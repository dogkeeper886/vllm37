// K80 fork (#74, #109): int4 and int8 weight-only kernels for Kepler (sm_37),
// in the layout K80Int4LinearKernel repacks every int4 and int8 checkpoint
// into (#73):
//   qweight int32 [N, K / P]  P = 32 / bits values per word along K, value i
//                             in bits (bits * i) and up
//   scales  fp32  [N, G]      group size K / G
//   zeros   fp32  [N, G]      w = (q - zero) * scale
// The K80 has no dp4a, half2 math or tensor cores, so both kernels decode a
// value by placing it in the mantissa of 2^23 with one byte permute (no
// integer-to-float conversion) and do the rest in FP32 (ollama37's
// mmvq-k80.cu).
//   k80_int4_gemm        y = x W^T for a few rows: one warp computes 4 or 8
//                        output features, each lane reads the packed words of
//                        all of them in one or two loads per step and shares
//                        its inputs across them from registers.
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

// The 32 / B values of a packed word, as 2^23 + value.
template <int B>
__device__ __forceinline__ void unpack(uint32_t q, float* v);
template <>
__device__ __forceinline__ void unpack<4>(uint32_t q, float* v) {
  // Byte j holds value 2j in its low nibble and 2j + 1 in its high one.
  const uint32_t lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
  v[0] = magic_byte<0>(lo); v[1] = magic_byte<0>(hi);
  v[2] = magic_byte<1>(lo); v[3] = magic_byte<1>(hi);
  v[4] = magic_byte<2>(lo); v[5] = magic_byte<2>(hi);
  v[6] = magic_byte<3>(lo); v[7] = magic_byte<3>(hi);
}
template <>
__device__ __forceinline__ void unpack<8>(uint32_t q, float* v) {
  v[0] = magic_byte<0>(q); v[1] = magic_byte<1>(q);
  v[2] = magic_byte<2>(q); v[3] = magic_byte<3>(q);
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
struct Words<8> {
  static __device__ __forceinline__ void load(const uint32_t* p, uint32_t* w) {
    const uint4 a = __ldg((const uint4*)p), b = __ldg((const uint4*)p + 1);
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
    w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
  }
};
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

// y[M, N] = x[M, K] W^T, B-bit weights.  A warp computes R output features
// for all M rows.  Lane t covers K = base .. base + E - 1, base = E (t + 32 s)
// at step s: it loads the E B / 32 words of each of the R features at once
// and the E inputs of each row once, so M * E inputs stay in registers (E =
// 32 up to 2 rows, 16 up to 4, then 8).  Sum (q - z) x = sum q x - z sum x,
// so the zero costs one FMA per E weights.  Needs K % E == 0 and group size
// % E == 0.
template <int R, int M, int E, int B, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    gemm_kernel(const T* __restrict__ x, const uint32_t* __restrict__ qw,
                const float* __restrict__ sc, const float* __restrict__ zr,
                T* __restrict__ y, int N, int K, int G, int gs) {
  constexpr int P = 32 / B;  // values per word
  constexpr int W = E / P;   // words per row per step
  const int lane = threadIdx.x;
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const int row_words = K / P;
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
      for (int i = 0; i < E / 8; ++i)
        Act<T>::load8(x + (size_t)m * K + base + 8 * i, xv[m] + 8 * i);
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < E; ++i) s += xv[m][i];
      xs[m] = s;
    }
    const int g = base / gs;
    uint32_t q[R][W];
    float s[R], z[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int rr = min(row0 + r, N - 1);
      Words<W>::load(qw + (size_t)rr * row_words + base / P, q[r]);
      s[r] = __ldg(sc + (size_t)rr * G + g);
      z[r] = __ldg(zr + (size_t)rr * G + g);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float d[M];
#pragma unroll
      for (int m = 0; m < M; ++m) d[m] = 0.0f;
#pragma unroll
      for (int i = 0; i < W; ++i) {
        float v[P];
        unpack<B>(q[r][i], v);
#pragma unroll
        for (int j = 0; j < P; ++j) v[j] -= 8388608.0f;
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < P; ++j) d[m] = fmaf(v[j], xv[m][P * i + j], d[m]);
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

template <int R, int M, int E, int B, typename T>
void launch_gemm(const T* x, const uint32_t* qw, const float* sc,
                 const float* zr, T* y, int N, int K, int G,
                 cudaStream_t stream) {
  const dim3 block(kWarp, kWarps), grid((N + R * kWarps - 1) / (R * kWarps));
  gemm_kernel<R, M, E, B, T>
      <<<grid, block, 0, stream>>>(x, qw, sc, zr, y, N, K, G, K / G);
}

// Rows m (1..8) with at most emax inputs per lane per step.  Measured on a
// K80 die (#74): E = 32 is fastest for 1-2 rows, 16 for 3-4, 8 above; 8
// features per warp beat 4 above 4 rows once N fills the die (N >= 2048).
template <int B, typename T>
void gemm_rows(int m, int emax, const T* x, const uint32_t* qw,
               const float* sc, const float* zr, T* y, int N, int K, int G,
               cudaStream_t s) {
#define K80_GEMM(R, M, E) \
  return launch_gemm<R, M, E, B, T>(x, qw, sc, zr, y, N, K, G, s)
  const int e = std::min(emax, m <= 2 ? 32 : m <= 4 ? 16 : 8);
  const bool wide = N >= 2048;
  switch (m) {
    case 1:
      if (e == 32) K80_GEMM(4, 1, 32);
      if (e == 16) K80_GEMM(4, 1, 16);
      K80_GEMM(4, 1, 8);
    case 2:
      if (e == 32) K80_GEMM(4, 2, 32);
      if (e == 16) K80_GEMM(4, 2, 16);
      K80_GEMM(4, 2, 8);
    case 3:
      if (e == 16) K80_GEMM(4, 3, 16);
      K80_GEMM(4, 3, 8);
    case 4:
      if (e == 16) K80_GEMM(4, 4, 16);
      K80_GEMM(4, 4, 8);
    case 5:
      if (wide) K80_GEMM(8, 5, 8);
      K80_GEMM(4, 5, 8);
    case 6:
      if (wide) K80_GEMM(8, 6, 8);
      K80_GEMM(4, 6, 8);
    case 7:
      if (wide) K80_GEMM(8, 7, 8);
      K80_GEMM(4, 7, 8);
    default:
      if (wide) K80_GEMM(8, 8, 8);
      K80_GEMM(4, 8, 8);
  }
#undef K80_GEMM
}

// Bit-identical to k80_int4_dequantize() in PyTorch.  FP32: (q - z) * s.
// FP16: z and s rounded to half, and a rounding after the subtract and after
// the multiply, as PyTorch's half ops do (the first is exact, and skipped,
// when z is a whole number of at most 1024).

// Values 4c .. 4c + 3 of a row of B-bit words, as 2^23 + value.
template <int B>
__device__ __forceinline__ void quad(const uint32_t* row, int c, float* v);
template <>
__device__ __forceinline__ void quad<4>(const uint32_t* row, int c, float* v) {
  const uint32_t q = __ldg(row + (c >> 1)) >> (16 * (c & 1));
  const uint32_t lo = q & 0x0F0Fu, hi = (q >> 4) & 0x0F0Fu;
  v[0] = magic_byte<0>(lo); v[1] = magic_byte<0>(hi);
  v[2] = magic_byte<1>(lo); v[3] = magic_byte<1>(hi);
}
template <>
__device__ __forceinline__ void quad<8>(const uint32_t* row, int c, float* v) {
  unpack<8>(__ldg(row + c), v);
}

// Values 8c .. 8c + 7 of a row of B-bit words, as 2^23 + value.
template <int B>
__device__ __forceinline__ void oct(const uint32_t* row, int c, float* v);
template <>
__device__ __forceinline__ void oct<4>(const uint32_t* row, int c, float* v) {
  unpack<4>(__ldg(row + c), v);
}
template <>
__device__ __forceinline__ void oct<8>(const uint32_t* row, int c, float* v) {
  const uint2 w = __ldg((const uint2*)row + c);
  unpack<8>(w.x, v);
  unpack<8>(w.y, v + 4);
}

// FP32, one thread per 4 values and one float4 store, so a warp's stores are
// contiguous.
template <int B>
__global__ void __launch_bounds__(256)
    dequant_f32_kernel(const uint32_t* __restrict__ qw,
                       const float* __restrict__ sc,
                       const float* __restrict__ zr, float* __restrict__ out,
                       int N, int K, int G, int gs) {
  const int quads = K / 4;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= quads) return;
  const int g = (c * 4) / gs;
  for (int r = blockIdx.y; r < N; r += gridDim.y) {
    float v[4];
    quad<B>(qw + (size_t)r * (K * B / 32), c, v);
    const float z = __ldg(zr + (size_t)r * G + g);
    const float s = __ldg(sc + (size_t)r * G + g);
    ((float4*)out)[(size_t)r * quads + c] = make_float4(
        __fmul_rn(__fsub_rn(v[0] - 8388608.0f, z), s),
        __fmul_rn(__fsub_rn(v[1] - 8388608.0f, z), s),
        __fmul_rn(__fsub_rn(v[2] - 8388608.0f, z), s),
        __fmul_rn(__fsub_rn(v[3] - 8388608.0f, z), s));
  }
}

__device__ __forceinline__ uint32_t half_bits(float v) {
  return (uint32_t)__half_as_ushort(__float2half_rn(v));
}

// FP16, one thread per 8 values: 8 halves, one 16-byte store.
template <int B>
__global__ void __launch_bounds__(256)
    dequant_f16_kernel(const uint32_t* __restrict__ qw,
                       const float* __restrict__ sc,
                       const float* __restrict__ zr, c10::Half* __restrict__ out,
                       int N, int K, int G, int gs) {
  const int octs = K / 8;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= octs) return;
  const int g = (c * 8) / gs;
  for (int r = blockIdx.y; r < N; r += gridDim.y) {
    float v[8];
    oct<B>(qw + (size_t)r * (K * B / 32), c, v);
    const float zh = __half2float(__float2half_rn(__ldg(zr + (size_t)r * G + g)));
    const float sh = __half2float(__float2half_rn(__ldg(sc + (size_t)r * G + g)));
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
    ((uint4*)out)[(size_t)r * octs + c] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// K for a layout check: qweight [N, K * bits / 32], scales and zeros [N, G].
int64_t check_layout(torch::Tensor const& qweight, torch::Tensor const& scales,
                     torch::Tensor const& zeros, int64_t bits) {
  TORCH_CHECK(bits == 4 || bits == 8, "k80_int4: bits must be 4 or 8");
  TORCH_CHECK(qweight.is_cuda() && qweight.dim() == 2 && qweight.is_contiguous() &&
                  qweight.scalar_type() == at::kInt,
              "k80_int4: qweight must be contiguous int32 [N, K * bits / 32]");
  TORCH_CHECK(scales.dim() == 2 && scales.is_contiguous() &&
                  scales.scalar_type() == at::kFloat &&
                  scales.size(0) == qweight.size(0),
              "k80_int4: scales must be contiguous float32 [N, G]");
  TORCH_CHECK(zeros.sizes() == scales.sizes() && zeros.is_contiguous() &&
                  zeros.scalar_type() == at::kFloat,
              "k80_int4: zeros must be contiguous float32 [N, G]");
  const int64_t K = qweight.size(1) * (32 / bits), G = scales.size(1);
  TORCH_CHECK(G > 0 && K % G == 0 && (K / G) % 8 == 0,
              "k80_int4: the group size must divide K and be a multiple of 8");
  TORCH_CHECK(qweight.size(0) <= INT32_MAX && K <= INT32_MAX,
              "k80_int4: weight too large");
  return K;
}

template <int B>
void gemm_bits(torch::Tensor const& x, torch::Tensor const& qweight,
               torch::Tensor const& scales, torch::Tensor const& zeros,
               torch::Tensor& y) {
  const int M = x.size(0), N = qweight.size(0), K = x.size(1),
            G = scales.size(1), gs = K / G;
  const int emax = (K % 32 == 0 && gs % 32 == 0)   ? 32
                   : (K % 16 == 0 && gs % 16 == 0) ? 16
                                                   : 8;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* qw = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  for (int m0 = 0; m0 < M; m0 += kMaxRows) {
    const int m = std::min(kMaxRows, M - m0);
    if (x.scalar_type() == at::kFloat)
      gemm_rows<B, float>(m, emax, x.data_ptr<float>() + (size_t)m0 * K, qw,
                          scales.data_ptr<float>(), zeros.data_ptr<float>(),
                          y.data_ptr<float>() + (size_t)m0 * N, N, K, G, stream);
    else
      gemm_rows<B, c10::Half>(m, emax, x.data_ptr<c10::Half>() + (size_t)m0 * K,
                              qw, scales.data_ptr<float>(),
                              zeros.data_ptr<float>(),
                              y.data_ptr<c10::Half>() + (size_t)m0 * N, N, K, G,
                              stream);
  }
}

template <int B>
void dequant_bits(torch::Tensor& out, torch::Tensor const& qweight,
                  torch::Tensor const& scales, torch::Tensor const& zeros) {
  const int N = out.size(0), K = out.size(1), G = scales.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* qw = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  const int rows = std::min(N, 65535);
  if (out.scalar_type() == at::kFloat) {
    const dim3 grid((K / 4 + 255) / 256, rows);
    dequant_f32_kernel<B><<<grid, 256, 0, stream>>>(
        qw, scales.data_ptr<float>(), zeros.data_ptr<float>(),
        out.data_ptr<float>(), N, K, G, K / G);
  } else {
    TORCH_CHECK(out.scalar_type() == at::kHalf,
                "k80_int4_dequantize: out must be float32 or float16");
    const dim3 grid((K / 8 + 255) / 256, rows);
    dequant_f16_kernel<B><<<grid, 256, 0, stream>>>(
        qw, scales.data_ptr<float>(), zeros.data_ptr<float>(),
        out.data_ptr<c10::Half>(), N, K, G, K / G);
  }
}

}  // namespace k80_int4

// y [M, N] = x [M, K] @ W^T, for any M (launched 8 rows at a time; past
// about 48 rows dequantize + cuBLAS is faster, #74).
torch::Tensor k80_int4_gemm(torch::Tensor const& x, torch::Tensor const& qweight,
                            torch::Tensor const& scales,
                            torch::Tensor const& zeros, int64_t bits) {
  using namespace k80_int4;
  const int64_t K = check_layout(qweight, scales, zeros, bits);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K && x.is_contiguous() &&
                  x.device() == qweight.device(),
              "k80_int4_gemm: x must be contiguous [M, K] on the weight's device");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "k80_int4_gemm: x must be float32 or float16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "k80_int4_gemm: x must be 16-byte aligned");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto y = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.size(0) == 0) return y;
  if (bits == 4)
    gemm_bits<4>(x, qweight, scales, zeros, y);
  else
    gemm_bits<8>(x, qweight, scales, zeros, y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// out [N, K] (float32 or float16) = the dequantized weight.
void k80_int4_dequantize(torch::Tensor& out, torch::Tensor const& qweight,
                         torch::Tensor const& scales,
                         torch::Tensor const& zeros, int64_t bits) {
  using namespace k80_int4;
  const int64_t K = check_layout(qweight, scales, zeros, bits);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == qweight.size(0) &&
                  out.size(1) == K && out.is_contiguous() &&
                  out.device() == qweight.device(),
              "k80_int4_dequantize: out must be contiguous [N, K]");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  if (out.size(0) == 0 || K == 0) return;
  if (bits == 4)
    dequant_bits<4>(out, qweight, scales, zeros);
  else
    dequant_bits<8>(out, qweight, scales, zeros);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
