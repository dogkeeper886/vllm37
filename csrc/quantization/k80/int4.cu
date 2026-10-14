// K80 fork (#74): int4 weight-only kernels for Kepler (sm_37), in the layout
// K80Int4LinearKernel repacks every int4 checkpoint into (#73):
//   qweight int32 [N, K / 8]  value i of a word in bits 4i..4i+3, along K
//   scales  fp32  [N, G]      group size K / G
//   zeros   fp32  [N, G]      w = (q - zero) * scale
// The K80 has no dp4a, half2 math or tensor cores, so both kernels decode a
// value by placing it in the mantissa of 2^23 with one byte permute (no
// integer-to-float conversion) and do the rest in FP32 (ollama37's
// mmvq-k80.cu).
//   k80_int4_gemm        y = x W^T for a few rows: one warp computes 4 or 8
//                        output features, each lane reads the packed words of
//                        all of them in one load per step and shares its
//                        inputs across them from registers.
//   k80_int4_dequantize  W in FP32 or FP16 for cuBLAS, bit for bit what the
//                        PyTorch reference k80_int4_dequantize() gives.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace k80_int4 {

constexpr int kWarp = 32;
constexpr int kWarps = 4;    // warps per block
constexpr int kMaxRows = 8;  // inputs per launch

// Byte k of q in the mantissa of 2^23, as a float: 2^23 + byte.
template <int k>
__device__ __forceinline__ float magic_byte(uint32_t q) {
  return __int_as_float(__byte_perm(q, 0x4B000000u, 0x7440 + k));
}

// 8 activations (16-byte aligned) as floats; one output from a float.
template <typename T>
struct Act;
template <>
struct Act<float> {
  static __device__ __forceinline__ void load8(const float* p, float* x) {
    const float4 a = __ldg((const float4*)p), b = __ldg((const float4*)p + 1);
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
    x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
  }
  static __device__ __forceinline__ void store(float* p, float v) { *p = v; }
};
template <>
struct Act<c10::Half> {
  static __device__ __forceinline__ void load8(const c10::Half* p, float* x) {
    const uint4 v = __ldg((const uint4*)p);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[2 * j] = __half2float(__ushort_as_half((unsigned short)(w[j] & 0xFFFF)));
      x[2 * j + 1] = __half2float(__ushort_as_half((unsigned short)(w[j] >> 16)));
    }
  }
  static __device__ __forceinline__ void store(c10::Half* p, float v) {
    *(__half*)p = __float2half_rn(v);
  }
};

// V consecutive packed words of a row, in one load.
template <int V>
struct Words;
template <>
struct Words<4> {
  static __device__ __forceinline__ void load(const uint32_t* p, uint32_t* w) {
    const uint4 v = __ldg((const uint4*)p);
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
  }
};
template <>
struct Words<2> {
  static __device__ __forceinline__ void load(const uint32_t* p, uint32_t* w) {
    const uint2 v = __ldg((const uint2*)p);
    w[0] = v.x; w[1] = v.y;
  }
};
template <>
struct Words<1> {
  static __device__ __forceinline__ void load(const uint32_t* p, uint32_t* w) {
    w[0] = __ldg(p);
  }
};

// y[M, N] = x[M, K] W^T.  A warp computes R output features for all M rows.
// Lane t covers K = base .. base + 8V - 1, base = 8V (t + 32 s) at step s: it
// loads V words of each of the R features at once and the 8V inputs of each
// row once, so M * 8V inputs stay in registers (V = 4 up to 2 rows, 2 up to
// 4, then 1).  Sum (q - z) x = sum q x - z sum x, so the zero costs one FMA
// per 8V weights.  Needs K % 8V == 0 and group size % 8V == 0.
template <int R, int M, int V, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    gemm_kernel(const T* __restrict__ x, const uint32_t* __restrict__ qw,
                const float* __restrict__ sc, const float* __restrict__ zr,
                T* __restrict__ y, int N, int K, int G, int gs) {
  constexpr int E = 8 * V;
  const int lane = threadIdx.x;
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const int row_words = K / 8;
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

  for (int base = lane * E; base < K; base += E * kWarp) {
    float xv[M][E], xs[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int i = 0; i < V; ++i)
        Act<T>::load8(x + (size_t)m * K + base + 8 * i, xv[m] + 8 * i);
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < E; ++i) s += xv[m][i];
      xs[m] = s;
    }
    const int g = base / gs;
    uint32_t q[R][V];
    float s[R], z[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int rr = min(row0 + r, N - 1);
      Words<V>::load(qw + (size_t)rr * row_words + base / 8, q[r]);
      s[r] = __ldg(sc + (size_t)rr * G + g);
      z[r] = __ldg(zr + (size_t)rr * G + g);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float d[M];
#pragma unroll
      for (int m = 0; m < M; ++m) d[m] = 0.0f;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        // Byte j holds value 2j in its low nibble and 2j + 1 in its high one.
        const uint32_t lo = q[r][i] & 0x0F0F0F0Fu;
        const uint32_t hi = (q[r][i] >> 4) & 0x0F0F0F0Fu;
        const float v[8] = {
            magic_byte<0>(lo) - 8388608.0f, magic_byte<0>(hi) - 8388608.0f,
            magic_byte<1>(lo) - 8388608.0f, magic_byte<1>(hi) - 8388608.0f,
            magic_byte<2>(lo) - 8388608.0f, magic_byte<2>(hi) - 8388608.0f,
            magic_byte<3>(lo) - 8388608.0f, magic_byte<3>(hi) - 8388608.0f};
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < 8; ++j) d[m] = fmaf(v[j], xv[m][8 * i + j], d[m]);
      }
#pragma unroll
      for (int m = 0; m < M; ++m)
        acc[r][m] = fmaf(s[r], fmaf(-z[r], xs[m], d[m]), acc[r][m]);
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == 0 && row0 + r < N)
        Act<T>::store(y + (size_t)m * N + row0 + r, v);
    }
}

template <int R, int M, int V, typename T>
void launch_gemm(const T* x, const uint32_t* qw, const float* sc,
                 const float* zr, T* y, int N, int K, int G,
                 cudaStream_t stream) {
  const dim3 block(kWarp, kWarps), grid((N + R * kWarps - 1) / (R * kWarps));
  gemm_kernel<R, M, V, T>
      <<<grid, block, 0, stream>>>(x, qw, sc, zr, y, N, K, G, K / G);
}

// Rows m (1..8) with at most vmax words per load.  Measured on a K80 die
// (#74): V = 4 is fastest for 1-2 rows, 2 for 3-4, 1 above; 8 features per
// warp beat 4 above 4 rows once N fills the die (N >= 2048).
template <typename T>
void gemm_rows(int m, int vmax, const T* x, const uint32_t* qw, const float* sc,
               const float* zr, T* y, int N, int K, int G, cudaStream_t s) {
#define K80_GEMM(R, M, V) \
  return launch_gemm<R, M, V, T>(x, qw, sc, zr, y, N, K, G, s)
  const int v = std::min(vmax, m <= 2 ? 4 : m <= 4 ? 2 : 1);
  const bool wide = N >= 2048;
  switch (m) {
    case 1:
      if (v == 4) K80_GEMM(4, 1, 4);
      if (v == 2) K80_GEMM(4, 1, 2);
      K80_GEMM(4, 1, 1);
    case 2:
      if (v == 4) K80_GEMM(4, 2, 4);
      if (v == 2) K80_GEMM(4, 2, 2);
      K80_GEMM(4, 2, 1);
    case 3:
      if (v == 2) K80_GEMM(4, 3, 2);
      K80_GEMM(4, 3, 1);
    case 4:
      if (v == 2) K80_GEMM(4, 4, 2);
      K80_GEMM(4, 4, 1);
    case 5:
      if (wide) K80_GEMM(8, 5, 1);
      K80_GEMM(4, 5, 1);
    case 6:
      if (wide) K80_GEMM(8, 6, 1);
      K80_GEMM(4, 6, 1);
    case 7:
      if (wide) K80_GEMM(8, 7, 1);
      K80_GEMM(4, 7, 1);
    default:
      if (wide) K80_GEMM(8, 8, 1);
      K80_GEMM(4, 8, 1);
  }
#undef K80_GEMM
}

// Bit-identical to k80_int4_dequantize() in PyTorch.  FP32: (q - z) * s.
// FP16: z and s rounded to half, and a rounding after the subtract and after
// the multiply, as PyTorch's half ops do (the first is exact, and skipped,
// when z is a whole number of at most 1024).
//
// FP32, two threads per word: thread t decodes half (t & 1) of word t / 2 and
// stores one float4, so a warp's stores are contiguous.
__global__ void __launch_bounds__(256)
    dequant_f32_kernel(const uint32_t* __restrict__ qw,
                       const float* __restrict__ sc,
                       const float* __restrict__ zr, float* __restrict__ out,
                       int N, int K, int G, int gs) {
  const int halves = K / 4;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= halves) return;
  const int g = (c * 4) / gs;
  for (int r = blockIdx.y; r < N; r += gridDim.y) {
    const size_t i = (size_t)r * halves + c;
    const uint32_t q = __ldg(qw + (i >> 1)) >> (16 * (c & 1));
    const float z = __ldg(zr + (size_t)r * G + g);
    const float s = __ldg(sc + (size_t)r * G + g);
    const uint32_t lo = q & 0x0F0Fu, hi = (q >> 4) & 0x0F0Fu;
    ((float4*)out)[i] = make_float4(
        __fmul_rn(__fsub_rn(magic_byte<0>(lo) - 8388608.0f, z), s),
        __fmul_rn(__fsub_rn(magic_byte<0>(hi) - 8388608.0f, z), s),
        __fmul_rn(__fsub_rn(magic_byte<1>(lo) - 8388608.0f, z), s),
        __fmul_rn(__fsub_rn(magic_byte<1>(hi) - 8388608.0f, z), s));
  }
}

__device__ __forceinline__ uint32_t half_bits(float v) {
  return (uint32_t)__half_as_ushort(__float2half_rn(v));
}

// FP16, one thread per word: 8 halves, one 16-byte store.
__global__ void __launch_bounds__(256)
    dequant_f16_kernel(const uint32_t* __restrict__ qw,
                       const float* __restrict__ sc,
                       const float* __restrict__ zr, c10::Half* __restrict__ out,
                       int N, int K, int G, int gs) {
  const int words = K / 8;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= words) return;
  const int g = (c * 8) / gs;
  for (int r = blockIdx.y; r < N; r += gridDim.y) {
    const size_t i = (size_t)r * words + c;
    const uint32_t q = __ldg(qw + i);
    const float zh = __half2float(__float2half_rn(__ldg(zr + (size_t)r * G + g)));
    const float sh = __half2float(__float2half_rn(__ldg(sc + (size_t)r * G + g)));
    const uint32_t lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
    const float v[8] = {magic_byte<0>(lo), magic_byte<0>(hi), magic_byte<1>(lo),
                        magic_byte<1>(hi), magic_byte<2>(lo), magic_byte<2>(hi),
                        magic_byte<3>(lo), magic_byte<3>(hi)};
    uint32_t o[4];
    if (zh == truncf(zh) && fabsf(zh) <= 1024.0f) {
      // q - z is a whole number below 2048, exact in half; 2^23 + z is exact.
      const float bias = 8388608.0f + zh;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        o[j] = half_bits(__fmul_rn(v[2 * j] - bias, sh)) |
               (half_bits(__fmul_rn(v[2 * j + 1] - bias, sh)) << 16);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a = __half2float(
            __float2half_rn(__fsub_rn(v[2 * j] - 8388608.0f, zh)));
        const float b = __half2float(
            __float2half_rn(__fsub_rn(v[2 * j + 1] - 8388608.0f, zh)));
        o[j] = half_bits(__fmul_rn(a, sh)) | (half_bits(__fmul_rn(b, sh)) << 16);
      }
    }
    ((uint4*)out)[i] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

void check_layout(torch::Tensor const& qweight, torch::Tensor const& scales,
                  torch::Tensor const& zeros) {
  TORCH_CHECK(qweight.is_cuda() && qweight.dim() == 2 && qweight.is_contiguous() &&
                  qweight.scalar_type() == at::kInt,
              "k80_int4: qweight must be contiguous int32 [N, K / 8]");
  TORCH_CHECK(scales.dim() == 2 && scales.is_contiguous() &&
                  scales.scalar_type() == at::kFloat &&
                  scales.size(0) == qweight.size(0),
              "k80_int4: scales must be contiguous float32 [N, G]");
  TORCH_CHECK(zeros.sizes() == scales.sizes() && zeros.is_contiguous() &&
                  zeros.scalar_type() == at::kFloat,
              "k80_int4: zeros must be contiguous float32 [N, G]");
  const int64_t K = qweight.size(1) * 8, G = scales.size(1);
  TORCH_CHECK(G > 0 && K % G == 0 && (K / G) % 8 == 0,
              "k80_int4: the group size must divide K and be a multiple of 8");
  TORCH_CHECK(qweight.size(0) <= INT32_MAX && K <= INT32_MAX,
              "k80_int4: weight too large");
}

}  // namespace k80_int4

// y [M, N] = x [M, K] @ W^T, for any M (launched 8 rows at a time; past
// about 48 rows dequantize + cuBLAS is faster, #74).
torch::Tensor k80_int4_gemm(torch::Tensor const& x, torch::Tensor const& qweight,
                            torch::Tensor const& scales,
                            torch::Tensor const& zeros) {
  using namespace k80_int4;
  check_layout(qweight, scales, zeros);
  const int N = qweight.size(0), K = qweight.size(1) * 8, G = scales.size(1);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K && x.is_contiguous() &&
                  x.device() == qweight.device(),
              "k80_int4_gemm: x must be contiguous [M, K] on the weight's device");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "k80_int4_gemm: x must be float32 or float16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "k80_int4_gemm: x must be 16-byte aligned");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const int M = x.size(0);
  auto y = torch::empty({M, N}, x.options());
  if (M == 0) return y;
  const int gs = K / G;
  const int vmax = (K % 32 == 0 && gs % 32 == 0)   ? 4
                   : (K % 16 == 0 && gs % 16 == 0) ? 2
                                                   : 1;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* qw = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  for (int m0 = 0; m0 < M; m0 += kMaxRows) {
    const int m = std::min(kMaxRows, M - m0);
    if (x.scalar_type() == at::kFloat)
      gemm_rows<float>(m, vmax, x.data_ptr<float>() + (size_t)m0 * K, qw,
                       scales.data_ptr<float>(), zeros.data_ptr<float>(),
                       y.data_ptr<float>() + (size_t)m0 * N, N, K, G, stream);
    else
      gemm_rows<c10::Half>(m, vmax, x.data_ptr<c10::Half>() + (size_t)m0 * K,
                           qw, scales.data_ptr<float>(), zeros.data_ptr<float>(),
                           y.data_ptr<c10::Half>() + (size_t)m0 * N, N, K, G,
                           stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// out [N, K] (float32 or float16) = the dequantized weight.
void k80_int4_dequantize(torch::Tensor& out, torch::Tensor const& qweight,
                         torch::Tensor const& scales,
                         torch::Tensor const& zeros) {
  using namespace k80_int4;
  check_layout(qweight, scales, zeros);
  const int N = qweight.size(0), K = qweight.size(1) * 8, G = scales.size(1);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == N && out.size(1) == K &&
                  out.is_contiguous() && out.device() == qweight.device(),
              "k80_int4_dequantize: out must be contiguous [N, K]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  if (N == 0 || K == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* qw = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  const int rows = std::min(N, 65535);
  if (out.scalar_type() == at::kFloat) {
    const dim3 grid((K / 4 + 255) / 256, rows);
    dequant_f32_kernel<<<grid, 256, 0, stream>>>(
        qw, scales.data_ptr<float>(), zeros.data_ptr<float>(),
        out.data_ptr<float>(), N, K, G, K / G);
  } else {
    TORCH_CHECK(out.scalar_type() == at::kHalf,
                "k80_int4_dequantize: out must be float32 or float16");
    const dim3 grid((K / 8 + 255) / 256, rows);
    dequant_f16_kernel<<<grid, 256, 0, stream>>>(
        qw, scales.data_ptr<float>(), zeros.data_ptr<float>(),
        out.data_ptr<c10::Half>(), N, K, G, K / G);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
