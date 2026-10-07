// K80 int4 weight-only GEMV prototype: y[N] = W[N,K] * x[K]
// W packed 8 x uint4 per uint32 along K; per-group (G=128) FP32 scale and zero.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cublas_v2.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int G = 128;

// MODE 0: magic-number unpack (OR into float mantissa, FP32 subtract). MODE 1: int->float convert.
template<int MODE>
__global__ void gemv_int4(const uint32_t* __restrict__ W, const float* __restrict__ scale,
                          const float* __restrict__ zero, const float* __restrict__ x,
                          float* __restrict__ y, int N, int K) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int row = blockIdx.x * (blockDim.x / 32) + warp;
  if (row >= N) return;
  const int words = K / 8;
  const uint32_t* wr = W + (size_t)row * words;
  float acc = 0.f;
  for (int w = lane; w < words; w += 32) {          // coalesced: lane i reads word i
    uint32_t q = __ldg(wr + w);
    int k0 = w * 8, g = k0 / G;
    float s = __ldg(scale + (size_t)row * (K / G) + g);
    float z = __ldg(zero  + (size_t)row * (K / G) + g);
    float part = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t nib = (q >> (4 * j)) & 0xF;
      float v;
      if (MODE == 0) v = __int_as_float(0x4B000000u | nib) - (8388608.0f + z); // 2^23 + nib - (2^23 + z)
      else           v = (float)nib - z;
      part = fmaf(v, xs[k0 + j], part);
    }
    acc = fmaf(part, s, acc);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[row] = acc;
}

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 4096, K = argc > 2 ? atoi(argv[2]) : 4096;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("%s sm_%d%d  N=%d K=%d\n", p.name, p.major, p.minor, N, K);
  size_t words = (size_t)N * K / 8, ng = (size_t)N * K / G;
  std::vector<uint32_t> hW(words); std::vector<float> hs(ng), hz(ng), hx(K), hWf((size_t)N * K);
  srand(1);
  for (auto& v : hW) v = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
  for (size_t i = 0; i < ng; ++i) { hs[i] = 0.01f + 0.001f * (rand() % 100); hz[i] = (float)(rand() % 16); }
  for (auto& v : hx) v = (rand() % 2000) / 1000.f - 1.f;
  for (int r = 0; r < N; ++r) for (int k = 0; k < K; ++k) {
    uint32_t nib = (hW[(size_t)r * (K / 8) + k / 8] >> (4 * (k % 8))) & 0xF;
    size_t gi = (size_t)r * (K / G) + k / G;
    hWf[(size_t)k * N + r] = ((float)nib - hz[gi]) * hs[gi];   // column-major for cuBLAS
  }
  uint32_t* dW; float *ds, *dz, *dx, *dy, *dWf, *dy2;
  CK(cudaMalloc(&dW, words * 4)); CK(cudaMalloc(&ds, ng * 4)); CK(cudaMalloc(&dz, ng * 4));
  CK(cudaMalloc(&dx, K * 4)); CK(cudaMalloc(&dy, N * 4)); CK(cudaMalloc(&dy2, N * 4));
  CK(cudaMalloc(&dWf, (size_t)N * K * 4));
  CK(cudaMemcpy(dW, hW.data(), words * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ds, hs.data(), ng * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dz, hz.data(), ng * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dx, hx.data(), K * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWf, hWf.data(), (size_t)N * K * 4, cudaMemcpyHostToDevice));
  cublasHandle_t h; cublasCreate(&h);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int it = 200, threads = 256; dim3 grid((N + 7) / 8); size_t sm = K * 4;
  float one = 1, zero0 = 0, ms;
  // cuBLAS FP32 GEMV reference
  cublasSgemv(h, CUBLAS_OP_N, N, K, &one, dWf, N, dx, 1, &zero0, dy2, 1);
  cudaEventRecord(a); for (int i = 0; i < it; ++i) cublasSgemv(h, CUBLAS_OP_N, N, K, &one, dWf, N, dx, 1, &zero0, dy2, 1);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
  double t_f32 = ms / it * 1e3; double bytes_f32 = (double)N * K * 4;
  printf("cuBLAS Sgemv FP32      : %8.1f us  %6.1f GB/s weight read\n", t_f32, bytes_f32 / t_f32 / 1e3);
  std::vector<float> ref(N); CK(cudaMemcpy(ref.data(), dy2, N * 4, cudaMemcpyDeviceToHost));
  auto run = [&](auto kern, const char* name) {
    kern<<<grid, threads, sm>>>(dW, ds, dz, dx, dy, N, K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i = 0; i < it; ++i) kern<<<grid, threads, sm>>>(dW, ds, dz, dx, dy, N, K);
    cudaEventRecord(b); cudaEventSynchronize(b); float m; cudaEventElapsedTime(&m, a, b);
    double t = m / it * 1e3, bytes = words * 4.0 + ng * 8.0;
    std::vector<float> out(N); CK(cudaMemcpy(out.data(), dy, N * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0; for (int i = 0; i < N; ++i) maxrel = fmax(maxrel, fabs(out[i] - ref[i]) / (fabs(ref[i]) + 1e-3));
    printf("%-23s: %8.1f us  %6.1f GB/s packed read  %.2fx vs FP32  max rel err %.1e\n", name, t, bytes / t / 1e3, t_f32 / t, maxrel);
  };
  run(gemv_int4<0>, "int4 magic-number unpack");
  run(gemv_int4<1>, "int4 int->float unpack");
}
