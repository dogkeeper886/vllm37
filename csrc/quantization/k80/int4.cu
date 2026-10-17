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
//   k80_int4_moe_gemm    the same product for MoE experts (#119): weights
//                        [E, N, K / P], and each block takes its expert and
//                        its input rows from device tensors, so a step needs
//                        no host sync.
//   k80_moe_dense_gemm   the pair mode of k80_int4_moe_gemm for FP32 or FP16
//                        expert weights [E, N, K], read in place (#119).

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
constexpr int kMoeBlock = 8; // slots per block row of moe_gemm_kernel, the
                             // block size of moe_align_block_size

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

// Where row m of the input is read and its output written: rows m of x and
// y (DenseRows), or per-slot pointers with padding slots skipped (SlotRows).
template <typename T>
struct DenseRows {
  const T* x;
  T* y;
  int K, N;
  __device__ __forceinline__ const T* in(int m) const { return x + (size_t)m * K; }
  __device__ __forceinline__ T* out(int m) const { return y + (size_t)m * N; }
  __device__ __forceinline__ bool valid(int) const { return true; }
};
template <typename T, int M>
struct SlotRows {
  const T* x[M];
  T* y[M];
  __device__ __forceinline__ const T* in(int m) const { return x[m]; }
  __device__ __forceinline__ T* out(int m) const { return y[m]; }
  __device__ __forceinline__ bool valid(int m) const { return y[m] != nullptr; }
};

// y[M, N] = x[M, K] W^T, B-bit weights, for the M rows of `rows`.  A warp
// computes R output features from row0 for all M rows.  Lane t covers K = base .. base + E - 1, base = E (t + 32 s)
// at step s: it loads the E B / 32 words of each of the R features at once
// and the E inputs of each row once, so M * E inputs stay in registers (E =
// 32 up to 2 rows, 16 up to 4, then 8).  Sum (q - z) x = sum q x - z sum x,
// so the zero costs one FMA per E weights.  Needs K % E == 0 and group size
// % E == 0.
template <int R, int M, int E, int B, typename T, typename Rows>
__device__ __forceinline__ void gemm_tile(const Rows& rows,
                                          const uint32_t* __restrict__ qw,
                                          const float* __restrict__ sc,
                                          const float* __restrict__ zr,
                                          int row0, int N, int K, int G,
                                          int gs) {
  constexpr int P = 32 / B;  // values per word
  constexpr int W = E / P;   // words per row per step
  const int lane = threadIdx.x;
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
        Act<T>::load8(rows.in(m) + base + 8 * i, xv[m] + 8 * i);
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
      if (lane == 0 && row0 + r < N && rows.valid(m))
        Act<T>::store(rows.out(m) + row0 + r, v);
    }
}

template <int R, int M, int E, int B, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    gemm_kernel(const T* __restrict__ x, const uint32_t* __restrict__ qw,
                const float* __restrict__ sc, const float* __restrict__ zr,
                T* __restrict__ y, int N, int K, int G, int gs) {
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const DenseRows<T> rows{x, y, K, N};
  gemm_tile<R, M, E, B, T>(rows, qw, sc, zr, row0, N, K, G, gs);
}

// MoE experts: weights [E, N, K / P] and scales, zeros [E, N, G].  Block row
// b (blockIdx.y) computes M slots of expert experts[b]: slot b in pair mode
// (slots null, M = 1), else slots[b M + m], where a slot of S or more is
// padding (moe_align_block_size's sorted_token_ids).  Slot s reads input row
// s / div and writes output row s.  Block rows from padded[0] / M on are
// empty.
template <int R, int M, int E, int B, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    moe_gemm_kernel(const T* __restrict__ x, const uint32_t* __restrict__ qw,
                    const float* __restrict__ sc, const float* __restrict__ zr,
                    T* __restrict__ y, const int* __restrict__ experts,
                    const int* __restrict__ slots,
                    const int* __restrict__ padded, int S, int div, int N,
                    int K, int G, int gs) {
  const int b = blockIdx.y;
  if (padded != nullptr && b * M >= __ldg(padded)) return;
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const int e = __ldg(experts + b);
  if (e < 0) return;
  SlotRows<T, M> rows;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const int s = slots != nullptr ? __ldg(slots + b * M + m) : b * M + m;
    const bool valid = s < S;
    rows.x[m] = x + (size_t)(valid ? s / div : 0) * K;
    rows.y[m] = valid ? y + (size_t)s * N : nullptr;
  }
  const size_t first = (size_t)e * N;  // the expert's first weight row
  gemm_tile<R, M, E, B, T>(rows, qw + first * (K / (32 / B)), sc + first * G,
                           zr + first * G, row0, N, K, G, gs);
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

template <int R, int M, int E, int B, typename T>
void launch_moe(const T* x, const uint32_t* qw, const float* sc,
                const float* zr, T* y, const int* experts, const int* slots,
                const int* padded, int blocks, int S, int div, int N, int K,
                int G, cudaStream_t stream) {
  const dim3 block(kWarp, kWarps),
      grid((N + R * kWarps - 1) / (R * kWarps), blocks);
  moe_gemm_kernel<R, M, E, B, T><<<grid, block, 0, stream>>>(
      x, qw, sc, zr, y, experts, slots, padded, S, div, N, K, G, K / G);
}

// Pair mode: one slot per block row, with the 1-row choice of gemm_rows.
// Block mode: kMoeBlock slots per block row, as gemm_rows does for 8 rows.
template <int B, typename T>
void moe_rows(bool pairs, int emax, const T* x, const uint32_t* qw,
              const float* sc, const float* zr, T* y, const int* experts,
              const int* slots, const int* padded, int blocks, int S, int div,
              int N, int K, int G, cudaStream_t s) {
#define K80_MOE(R, M, E)                                                     \
  return launch_moe<R, M, E, B, T>(x, qw, sc, zr, y, experts, slots, padded, \
                                   blocks, S, div, N, K, G, s)
  if (pairs) {
    if (emax == 32) K80_MOE(4, 1, 32);
    if (emax == 16) K80_MOE(4, 1, 16);
    K80_MOE(4, 1, 8);
  }
  if (N >= 2048) K80_MOE(8, kMoeBlock, 8);
  K80_MOE(4, kMoeBlock, 8);
#undef K80_MOE
}

template <int B>
void moe_bits(torch::Tensor const& x, torch::Tensor const& qweight,
              torch::Tensor const& scales, torch::Tensor const& zeros,
              torch::Tensor const& experts, const int* slots,
              const int* padded, int blocks, int S, int div, torch::Tensor& y) {
  const int N = qweight.size(1), K = x.size(1), G = scales.size(2),
            gs = K / G;
  const int emax = (K % 32 == 0 && gs % 32 == 0)   ? 32
                   : (K % 16 == 0 && gs % 16 == 0) ? 16
                                                   : 8;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* qw = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  const bool pairs = slots == nullptr;
  if (x.scalar_type() == at::kFloat)
    moe_rows<B, float>(pairs, emax, x.data_ptr<float>(), qw,
                       scales.data_ptr<float>(), zeros.data_ptr<float>(),
                       y.data_ptr<float>(), experts.data_ptr<int>(), slots,
                       padded, blocks, S, div, N, K, G, stream);
  else
    moe_rows<B, c10::Half>(pairs, emax, x.data_ptr<c10::Half>(), qw,
                           scales.data_ptr<float>(), zeros.data_ptr<float>(),
                           y.data_ptr<c10::Half>(), experts.data_ptr<int>(),
                           slots, padded, blocks, S, div, N, K, G, stream);
}

// y[s] = x[s / div] W[experts[s]]^T for FP32 or FP16 weights [E, N, K], one
// slot per block row.  Lane t reads 8 inputs and 8 weights of each of the R
// features at K = 8 (t + 32 i).  Needs K % 8 == 0.
template <int R, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    moe_dense_kernel(const T* __restrict__ x, const T* __restrict__ w,
                     T* __restrict__ y, const int* __restrict__ experts,
                     int div, int N, int K) {
  const int s = blockIdx.y;
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const int e = __ldg(experts + s);
  if (e < 0) return;
  const T* xr = x + (size_t)(s / div) * K;
  const T* wr = w + (size_t)e * N * K;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  for (int base = threadIdx.x * 8; base < K; base += 8 * kWarp) {
    float xv[8];
    Act<T>::load8(xr + base, xv);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float wv[8];
      Act<T>::load8(wr + (size_t)min(row0 + r, N - 1) * K + base, wv);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[r] = fmaf(wv[j], xv[j], acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    if (threadIdx.x == 0 && row0 + r < N)
      Act<T>::store(y + (size_t)s * N + row0 + r, v);
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

// y [S, N]: slot s is x [s / rows_per_input] @ W[e]^T, e its expert (#119).
// Pair mode (slots absent): expert_ids [S], one per slot.  Block mode:
// slots [8 b + m] and expert_ids [b] as moe_align_block_size gives them for
// block size 8, padded [0] the slots in use (padding slots are S or more).
torch::Tensor k80_int4_moe_gemm(torch::Tensor const& x,
                                torch::Tensor const& qweight,
                                torch::Tensor const& scales,
                                torch::Tensor const& zeros,
                                torch::Tensor const& expert_ids,
                                std::optional<torch::Tensor> const& slots,
                                std::optional<torch::Tensor> const& padded,
                                int64_t num_slots, int64_t rows_per_input,
                                int64_t bits) {
  using namespace k80_int4;
  TORCH_CHECK(bits == 4 || bits == 8, "k80_int4_moe_gemm: bits must be 4 or 8");
  TORCH_CHECK(qweight.is_cuda() && qweight.dim() == 3 &&
                  qweight.is_contiguous() && qweight.scalar_type() == at::kInt,
              "k80_int4_moe_gemm: qweight must be contiguous int32 "
              "[E, N, K * bits / 32]");
  TORCH_CHECK(scales.dim() == 3 && scales.is_contiguous() &&
                  scales.scalar_type() == at::kFloat &&
                  scales.size(0) == qweight.size(0) &&
                  scales.size(1) == qweight.size(1),
              "k80_int4_moe_gemm: scales must be contiguous float32 [E, N, G]");
  TORCH_CHECK(zeros.sizes() == scales.sizes() && zeros.is_contiguous() &&
                  zeros.scalar_type() == at::kFloat,
              "k80_int4_moe_gemm: zeros must be contiguous float32 [E, N, G]");
  const int64_t K = qweight.size(2) * (32 / bits), G = scales.size(2);
  TORCH_CHECK(G > 0 && K % G == 0 && (K / G) % 8 == 0,
              "k80_int4_moe_gemm: the group size must divide K and be a "
              "multiple of 8");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K && x.is_contiguous() &&
                  x.device() == qweight.device(),
              "k80_int4_moe_gemm: x must be contiguous [M, K] on the weight's "
              "device");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "k80_int4_moe_gemm: x must be float32 or float16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "k80_int4_moe_gemm: x must be 16-byte aligned");
  TORCH_CHECK(rows_per_input >= 1 &&
                  x.size(0) * rows_per_input >= num_slots,
              "k80_int4_moe_gemm: x has fewer rows than the slots read");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.is_contiguous() &&
                  expert_ids.scalar_type() == at::kInt &&
                  expert_ids.device() == qweight.device(),
              "k80_int4_moe_gemm: expert_ids must be contiguous int32");
  TORCH_CHECK(qweight.size(1) <= INT32_MAX && K <= INT32_MAX &&
                  num_slots <= INT32_MAX,
              "k80_int4_moe_gemm: too large");
  int blocks;
  const int* slot_ptr = nullptr;
  const int* padded_ptr = nullptr;
  if (slots.has_value()) {
    TORCH_CHECK(padded.has_value(),
                "k80_int4_moe_gemm: block mode needs padded");
    TORCH_CHECK(slots->dim() == 1 && slots->is_contiguous() &&
                    slots->scalar_type() == at::kInt &&
                    slots->numel() % kMoeBlock == 0,
                "k80_int4_moe_gemm: slots must be contiguous int32, a "
                "multiple of 8 long");
    TORCH_CHECK(padded->numel() >= 1 && padded->scalar_type() == at::kInt,
                "k80_int4_moe_gemm: padded must be int32");
    blocks = slots->numel() / kMoeBlock;
    TORCH_CHECK(expert_ids.numel() >= blocks,
                "k80_int4_moe_gemm: one expert id per block of 8 slots");
    slot_ptr = slots->data_ptr<int>();
    padded_ptr = padded->data_ptr<int>();
  } else {
    TORCH_CHECK(expert_ids.numel() == num_slots,
                "k80_int4_moe_gemm: one expert id per slot");
    blocks = num_slots;
  }
  TORCH_CHECK(blocks <= 65535, "k80_int4_moe_gemm: more than 65535 block rows");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto y = torch::empty({num_slots, qweight.size(1)}, x.options());
  if (num_slots == 0 || blocks == 0) return y;
  if (bits == 4)
    moe_bits<4>(x, qweight, scales, zeros, expert_ids, slot_ptr, padded_ptr,
                blocks, num_slots, rows_per_input, y);
  else
    moe_bits<8>(x, qweight, scales, zeros, expert_ids, slot_ptr, padded_ptr,
                blocks, num_slots, rows_per_input, y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// y [S, N]: slot s is x [s / rows_per_input] @ w[expert_ids[s]]^T, with FP32
// or FP16 weights w [E, N, K] in x's dtype (#119).
torch::Tensor k80_moe_dense_gemm(torch::Tensor const& x, torch::Tensor const& w,
                                 torch::Tensor const& expert_ids,
                                 int64_t rows_per_input) {
  using namespace k80_int4;
  TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.is_contiguous(),
              "k80_moe_dense_gemm: w must be contiguous [E, N, K]");
  const int64_t N = w.size(1), K = w.size(2), S = expert_ids.numel();
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K && x.is_contiguous() &&
                  x.scalar_type() == w.scalar_type() &&
                  x.device() == w.device(),
              "k80_moe_dense_gemm: x must be contiguous [M, K], w's dtype");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "k80_moe_dense_gemm: float32 or float16");
  TORCH_CHECK(K % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "k80_moe_dense_gemm: K must divide by 8, x and w 16-byte "
              "aligned");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.is_contiguous() &&
                  expert_ids.scalar_type() == at::kInt &&
                  expert_ids.device() == w.device(),
              "k80_moe_dense_gemm: expert_ids must be contiguous int32");
  TORCH_CHECK(rows_per_input >= 1 && x.size(0) * rows_per_input >= S,
              "k80_moe_dense_gemm: x has fewer rows than the slots read");
  TORCH_CHECK(S <= 65535 && N <= INT32_MAX && K <= INT32_MAX,
              "k80_moe_dense_gemm: too large");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto y = torch::empty({S, N}, x.options());
  if (S == 0) return y;
  constexpr int R = 4;
  const dim3 block(kWarp, kWarps), grid((N + R * kWarps - 1) / (R * kWarps), S);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kFloat)
    moe_dense_kernel<R, float><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        expert_ids.data_ptr<int>(), rows_per_input, N, K);
  else
    moe_dense_kernel<R, c10::Half><<<grid, block, 0, stream>>>(
        x.data_ptr<c10::Half>(), w.data_ptr<c10::Half>(),
        y.data_ptr<c10::Half>(), expert_ids.data_ptr<int>(), rows_per_input,
        N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
