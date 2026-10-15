// K80 fork (#111): GGUF Q4_K and Q6_K weights on Kepler (sm_37), read as
// stored, so a GGUF file takes its own size in VRAM. The weight is the GGUF
// tensor as loaded: uint8 [N, K / 256 * block bytes], one row per output
// feature.
//   Q4_K  144-byte blocks of 256: fp16 d and dmin, 12 bytes of 6-bit scales
//         and mins for 8 sub-blocks of 32, 128 bytes of 4-bit values.
//         w = d * sc * q - dmin * m.
//   Q6_K  210-byte blocks of 256: 128 bytes of low 4 bits, 64 of high 2 bits,
//         16 int8 scales (one per 16 values), fp16 d. w = d * sc * (q - 32).
// The shapes follow ollama37's mmvq-k80.cu and convert.cu for the same cards:
// values decoded by placing a byte in the mantissa of 2^23 with one byte
// permute, everything else in FP32 (no dp4a, half2 math or tensor cores).
//   k80_gguf_gemm        y = x W^T for a few rows: one warp computes 4 (Q4_K)
//                        or 2 (Q6_K) output features, each lane decoding its
//                        words once per step and applying them to every row
//                        of x. x is first copied to FP32, with the sums of
//                        each 16 inputs for Q4_K's min term.
//   k80_gguf_dequantize  W in FP32 or FP16 for cuBLAS, bit for bit what the
//                        gguf package's dequantize gives (rounded to FP16).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace k80_gguf {

constexpr int kWarp = 32;
constexpr int kWarps = 4;    // warps per block
constexpr int kMaxRows = 8;  // rows of x per launch
constexpr int kQK = 256;     // values per block
constexpr int kQ4KBytes = 144;
constexpr int kQ6KBytes = 210;
constexpr int64_t kTypeQ4K = 12;  // GGMLQuantizationType
constexpr int64_t kTypeQ6K = 14;

// Byte k of q in the mantissa of 2^23, as a float: 2^23 + byte.
template <int k>
__device__ __forceinline__ float magic_byte(uint32_t q) {
  return __int_as_float(__byte_perm(q, 0x4B000000u, 0x7440 + k));
}

// An integer below 2^23 as a float, exact, with no conversion instruction.
__device__ __forceinline__ float magic(uint32_t bits) {
  return __int_as_float(0x4B000000u | bits) - 8388608.0f;
}

__device__ __forceinline__ float half_at(const uint8_t* p) {
  return __half2float(__ushort_as_half(__ldg((const unsigned short*)p)));
}

// 4 bytes from a 2-byte-aligned address (a Q6_K block is 210 bytes).
__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  const unsigned short* q = (const unsigned short*)p;
  return (uint32_t)__ldg(q) | ((uint32_t)__ldg(q + 1) << 16);
}

// A Q6_K int8 scale as a float: flip the sign bit, decode, subtract the bias.
__device__ __forceinline__ float int8_at(const uint8_t* p) {
  return magic((uint32_t)__ldg(p) ^ 0x80u) - 128.0f;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  return v;
}

template <typename T>
__device__ __forceinline__ void store(T* p, float v);
template <>
__device__ __forceinline__ void store<float>(float* p, float v) { *p = v; }
template <>
__device__ __forceinline__ void store<c10::Half>(c10::Half* p, float v) {
  *(__half*)p = __float2half_rn(v);
}

// 4 consecutive outputs (16-byte aligned for float, 8 for half).
template <typename T>
__device__ __forceinline__ void store4(T* p, float a, float b, float c, float d);
template <>
__device__ __forceinline__ void store4<float>(float* p, float a, float b, float c,
                                              float d) {
  *(float4*)p = make_float4(a, b, c, d);
}
template <>
__device__ __forceinline__ void store4<c10::Half>(c10::Half* p, float a, float b,
                                                  float c, float d) {
  const uint32_t lo = (uint32_t)__half_as_ushort(__float2half_rn(a)) |
                      ((uint32_t)__half_as_ushort(__float2half_rn(b)) << 16);
  const uint32_t hi = (uint32_t)__half_as_ushort(__float2half_rn(c)) |
                      ((uint32_t)__half_as_ushort(__float2half_rn(d)) << 16);
  *(uint2*)p = make_uint2(lo, hi);
}

// x [M, K] -> ys [M, K] in FP32 and, when ysum16 is set, the sum of each 16
// consecutive inputs (Q4_K's min term). Grid (K / 256, M), 256 threads.
template <typename T>
__global__ void prep_kernel(const T* __restrict__ x, float* __restrict__ ys,
                            float* __restrict__ ysum16, int K) {
  const size_t m = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (float)x[m * K + i];
  ys[m * K + i] = v;
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o, 16);
  if (ysum16 != nullptr && (i & 15) == 0) ysum16[m * (K / 16) + (i >> 4)] = v;
}

// Q4_K, R output features per warp, M rows of x. Lane t reads 16 value bytes
// of block (b0 + t / 8): chunk j = (t % 8) / 2 at offset l = 16 * (t % 2).
// Low nibbles are values 64j + l.. (sub-block 2j), high nibbles 64j + 32 + l..
// (sub-block 2j + 1).
template <int R, int M, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    q4k_gemm_kernel(const uint8_t* __restrict__ w, const float* __restrict__ ys,
                    const float* __restrict__ ysum16, T* __restrict__ y, int N,
                    int K, size_t row_bytes) {
  const int lane = threadIdx.x;
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const int nb = K / kQK;
  const int bo = lane >> 3;
  const int j = (lane & 7) >> 1;
  const int l = (lane & 1) * 16;
  const int qoff = 16 + 32 * j + l;
  const bool upper = j >= 2;  // sub-blocks 4..7 split scale and min over bytes
  const int sh = 16 * (j & 1);

  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

  for (int b0 = 0; b0 < nb; b0 += 4) {
    const int b = b0 + bo;
    if (b >= nb) break;
    uint32_t q[R][4];
    float d1[R], d2[R], m1[R], m2[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint8_t* bp =
          w + (size_t)min(row0 + r, N - 1) * row_bytes + (size_t)b * kQ4KBytes;
      const uint4 hd = __ldg((const uint4*)bp);
      const uint4 qv = __ldg((const uint4*)(bp + qoff));
      const float d = __half2float(__ushort_as_half((unsigned short)(hd.x & 0xFFFF)));
      const float dmin = __half2float(__ushort_as_half((unsigned short)(hd.x >> 16)));
      const uint32_t a = hd.y >> sh, mb = hd.z >> sh, c = hd.w >> sh;
      const uint32_t scw = upper ? (c & 0x0F0Fu) | ((a >> 2) & 0x3030u) : a & 0x3F3Fu;
      const uint32_t mw =
          upper ? ((c >> 4) & 0x0F0Fu) | ((mb >> 2) & 0x3030u) : mb & 0x3F3Fu;
      d1[r] = d * magic(scw & 0xFFu);
      d2[r] = d * magic(scw >> 8);
      m1[r] = dmin * magic(mw & 0xFFu);
      m2[r] = dmin * magic(mw >> 8);
      q[r][0] = qv.x; q[r][1] = qv.y; q[r][2] = qv.z; q[r][3] = qv.w;
    }
    const int e = b * kQK + 64 * j + l;
    // The min term: dmin * m times the sum of the sub-block's inputs.
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float* sr = ysum16 + (size_t)m * (K / 16);
      const float sl = __ldg(sr + (e >> 4)), shs = __ldg(sr + ((e + 32) >> 4));
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r][m] -= m1[r] * sl + m2[r] * shs;
    }
    // Word v of every feature is decoded once and used for every row of x,
    // which loads only that word's 8 inputs.
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      float wl[R][4], wh[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t lo = q[r][v] & 0x0F0F0F0Fu, hi = (q[r][v] >> 4) & 0x0F0F0F0Fu;
        wl[r][0] = magic_byte<0>(lo) - 8388608.0f;
        wl[r][1] = magic_byte<1>(lo) - 8388608.0f;
        wl[r][2] = magic_byte<2>(lo) - 8388608.0f;
        wl[r][3] = magic_byte<3>(lo) - 8388608.0f;
        wh[r][0] = magic_byte<0>(hi) - 8388608.0f;
        wh[r][1] = magic_byte<1>(hi) - 8388608.0f;
        wh[r][2] = magic_byte<2>(hi) - 8388608.0f;
        wh[r][3] = magic_byte<3>(hi) - 8388608.0f;
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float* yr = ys + (size_t)m * K + e;
        const float4 yl = __ldg((const float4*)yr + v);
        const float4 yh = __ldg((const float4*)(yr + 32) + v);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float al = wl[r][0] * yl.x;
          al = fmaf(wl[r][1], yl.y, al);
          al = fmaf(wl[r][2], yl.z, al);
          al = fmaf(wl[r][3], yl.w, al);
          float ah = wh[r][0] * yh.x;
          ah = fmaf(wh[r][1], yh.y, ah);
          ah = fmaf(wh[r][2], yh.z, ah);
          ah = fmaf(wh[r][3], yh.w, ah);
          acc[r][m] = fmaf(d1[r], al, fmaf(d2[r], ah, acc[r][m]));
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0 && row0 + r < N) store(y + (size_t)m * N + row0 + r, v);
    }
}

// Q6_K, R output features per warp, M rows of x, 2 blocks per step. Lane t
// decodes, in block (b0 + t / 16) and half h = (t / 8) % 2, the values
// 128h + 32g + l0 + k for g, k in 0..3 with l0 = 4 (t % 8): low 4 bits from
// ql, high 2 bits from qh, rebuilt as the four bytes of q[g]. The "- 32" folds
// into each byte's decode constant.
template <int R, int M, typename T>
__global__ void __launch_bounds__(kWarp* kWarps)
    q6k_gemm_kernel(const uint8_t* __restrict__ w, const float* __restrict__ ys,
                    T* __restrict__ y, int N, int K, size_t row_bytes) {
  const int lane = threadIdx.x;
  const int row0 = (blockIdx.x * kWarps + threadIdx.y) * R;
  if (row0 >= N) return;
  const int nb = K / kQK;
  const int bo = lane >> 4;
  const int h = (lane >> 3) & 1;
  const int l0 = 4 * (lane & 7);
  const int ql_off = 64 * h + l0;
  const int qh_off = 128 + 32 * h + l0;
  const int sc_off = 192 + 8 * h + (l0 >> 4);

  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

  for (int b0 = 0; b0 < nb; b0 += 2) {
    const int b = b0 + bo;
    if (b >= nb) break;
    uint32_t q[R][4];
    float ds[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint8_t* bp =
          w + (size_t)min(row0 + r, N - 1) * row_bytes + (size_t)b * kQ6KBytes;
      const uint32_t qa = ld4(bp + ql_off);
      const uint32_t qb = ld4(bp + ql_off + 32);
      const uint32_t qh = ld4(bp + qh_off);
      const float d = half_at(bp + 208);
      q[r][0] = (qa & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
      q[r][1] = (qb & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2);
      q[r][2] = ((qa >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
      q[r][3] = ((qb >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u);
#pragma unroll
      for (int g = 0; g < 4; ++g) ds[r][g] = d * int8_at(bp + sc_off + 2 * g);
    }
    // Group g of every feature is decoded once and used for every row of x,
    // which loads only that group's 4 inputs.
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      float wq[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        // (2^23 + v) - (2^23 + 32) = v - 32, exact
        wq[r][0] = magic_byte<0>(q[r][g]) - 8388640.0f;
        wq[r][1] = magic_byte<1>(q[r][g]) - 8388640.0f;
        wq[r][2] = magic_byte<2>(q[r][g]) - 8388640.0f;
        wq[r][3] = magic_byte<3>(q[r][g]) - 8388640.0f;
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float4 yv = __ldg(
            (const float4*)(ys + (size_t)m * K + b * kQK + 128 * h + 32 * g + l0));
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float dot = wq[r][0] * yv.x;
          dot = fmaf(wq[r][1], yv.y, dot);
          dot = fmaf(wq[r][2], yv.z, dot);
          dot = fmaf(wq[r][3], yv.w, dot);
          acc[r][m] = fmaf(ds[r][g], dot, acc[r][m]);
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0 && row0 + r < N) store(y + (size_t)m * N + row0 + r, v);
    }
}

// Q4_K to FP32 or FP16: one warp per block, lane t decoding value word t (8
// values) and storing two groups of 4. Each step rounds as the gguf package's
// (d * sc) * q - dmin * m does.
template <typename T>
__global__ void __launch_bounds__(256)
    q4k_dequant_kernel(const uint8_t* __restrict__ w, T* __restrict__ out,
                       int64_t nblocks) {
  const int64_t i = (int64_t)blockIdx.x * 8 + threadIdx.x / kWarp;
  if (i >= nblocks) return;
  const int t = threadIdx.x % kWarp;
  const int j = t / 8;
  const uint8_t* bp = w + i * kQ4KBytes;
  const uint8_t* s = bp + 4;
  const uint32_t q = __ldg((const uint32_t*)(bp + 16) + t);
  const float d = half_at(bp), dmin = half_at(bp + 2);
  uint32_t sc1, m1, sc2, m2;  // ggml get_scale_min_k4 for sub-blocks 2j, 2j+1
  if (j < 2) {
    sc1 = __ldg(s + 2 * j) & 63;
    m1 = __ldg(s + 2 * j + 4) & 63;
    sc2 = __ldg(s + 2 * j + 1) & 63;
    m2 = __ldg(s + 2 * j + 5) & 63;
  } else {
    const int k = 2 * j;
    sc1 = (__ldg(s + k + 4) & 0xF) | ((__ldg(s + k - 4) >> 6) << 4);
    m1 = (__ldg(s + k + 4) >> 4) | ((__ldg(s + k) >> 6) << 4);
    sc2 = (__ldg(s + k + 5) & 0xF) | ((__ldg(s + k - 3) >> 6) << 4);
    m2 = (__ldg(s + k + 5) >> 4) | ((__ldg(s + k + 1) >> 6) << 4);
  }
  const float d1 = __fmul_rn(d, magic(sc1)), dm1 = __fmul_rn(dmin, magic(m1));
  const float d2 = __fmul_rn(d, magic(sc2)), dm2 = __fmul_rn(dmin, magic(m2));
  const uint32_t lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
#define K80_Q4K_V(dd, mm, w, k) \
  __fsub_rn(__fmul_rn(dd, magic_byte<k>(w) - 8388608.0f), mm)
  T* y = out + i * kQK + 64 * j + 4 * (t % 8);
  store4(y, K80_Q4K_V(d1, dm1, lo, 0), K80_Q4K_V(d1, dm1, lo, 1),
         K80_Q4K_V(d1, dm1, lo, 2), K80_Q4K_V(d1, dm1, lo, 3));
  store4(y + 32, K80_Q4K_V(d2, dm2, hi, 0), K80_Q4K_V(d2, dm2, hi, 1),
         K80_Q4K_V(d2, dm2, hi, 2), K80_Q4K_V(d2, dm2, hi, 3));
#undef K80_Q4K_V
}

// Q6_K to FP32 or FP16: 16 lanes per block. Lane t covers half ip = (t / 8)
// % 2 and the 4 values il = 4 (t % 8).. of each group of 32 in it; each value
// is (d * sc) * (q - 32), as the gguf package rounds it.
template <typename T>
__global__ void __launch_bounds__(256)
    q6k_dequant_kernel(const uint8_t* __restrict__ w, T* __restrict__ out,
                       int64_t nblocks) {
  const int64_t i = (int64_t)blockIdx.x * 16 + threadIdx.x / 16;
  if (i >= nblocks) return;
  const int t = threadIdx.x % 16;
  const int ip = t / 8;
  const int il = 4 * (t % 8);
  const uint8_t* bp = w + i * kQ6KBytes;
  const uint32_t ql0 = ld4(bp + 64 * ip + il);
  const uint32_t ql1 = ld4(bp + 64 * ip + il + 32);
  const uint32_t qh = ld4(bp + 128 + 32 * ip + il);
  const float d = half_at(bp + 208);
  const uint8_t* sc = bp + 192 + 8 * ip + il / 16;
  const uint32_t q[4] = {(ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4),
                         (ql1 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2),
                         ((ql0 >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u),
                         ((ql1 >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u)};
  T* y = out + i * kQK + 128 * ip + il;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const float ds = __fmul_rn(d, int8_at(sc + 2 * g));
    store4(y + 32 * g, __fmul_rn(ds, magic_byte<0>(q[g]) - 8388640.0f),
           __fmul_rn(ds, magic_byte<1>(q[g]) - 8388640.0f),
           __fmul_rn(ds, magic_byte<2>(q[g]) - 8388640.0f),
           __fmul_rn(ds, magic_byte<3>(q[g]) - 8388640.0f));
  }
}

// Features per warp: Q4_K 4 (ollama37's); Q6_K 2 for one row of x
// (ollama37's, arithmetic-bound), 4 from 2 rows up, where each decoded value
// serves more inputs.
constexpr int kQ4KFeatures = 4;
template <int M>
constexpr int kQ6KFeatures = M == 1 ? 2 : 4;

template <int M, typename T>
void gemm_m(int64_t type, const uint8_t* w, const float* ys, const float* ysum16,
            T* y, int N, int K, size_t row_bytes, cudaStream_t stream) {
  const dim3 block(kWarp, kWarps);
  if (type == kTypeQ4K) {
    constexpr int R = kQ4KFeatures;
    q4k_gemm_kernel<R, M, T><<<(N + R * kWarps - 1) / (R * kWarps), block, 0,
                               stream>>>(w, ys, ysum16, y, N, K, row_bytes);
  } else {
    constexpr int R = kQ6KFeatures<M>;
    q6k_gemm_kernel<R, M, T><<<(N + R * kWarps - 1) / (R * kWarps), block, 0,
                               stream>>>(w, ys, y, N, K, row_bytes);
  }
}

template <typename T>
void gemm_rows(int m, int64_t type, const uint8_t* w, const float* ys,
               const float* ysum16, T* y, int N, int K, size_t row_bytes,
               cudaStream_t stream) {
  switch (m) {
#define K80_GGUF_M(MM)                                                    \
  case MM:                                                                \
    gemm_m<MM, T>(type, w, ys, ysum16, y, N, K, row_bytes, stream); \
    break;
    K80_GGUF_M(1) K80_GGUF_M(2) K80_GGUF_M(3) K80_GGUF_M(4)
    K80_GGUF_M(5) K80_GGUF_M(6) K80_GGUF_M(7) K80_GGUF_M(8)
#undef K80_GGUF_M
  }
}

int block_bytes(int64_t type) {
  TORCH_CHECK(type == kTypeQ4K || type == kTypeQ6K,
              "k80_gguf: type must be Q4_K (12) or Q6_K (14), got ", type);
  return type == kTypeQ4K ? kQ4KBytes : kQ6KBytes;
}

// K of a GGUF weight [N, row bytes].
int64_t check_weight(torch::Tensor const& qweight, int64_t type) {
  const int bytes = block_bytes(type);
  TORCH_CHECK(qweight.is_cuda() && qweight.dim() == 2 && qweight.is_contiguous() &&
                  qweight.scalar_type() == at::kByte &&
                  qweight.size(1) % bytes == 0,
              "k80_gguf: qweight must be contiguous uint8 [N, K / 256 * ", bytes,
              "]");
  // Q4_K blocks are read as uint4; Q6_K blocks (210 bytes) 2 bytes at a time.
  const int align = type == kTypeQ4K ? 16 : 2;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(qweight.data_ptr()) % align == 0,
              "k80_gguf: qweight must be ", align, "-byte aligned");
  const int64_t K = qweight.size(1) / bytes * kQK;
  TORCH_CHECK(qweight.size(0) <= INT32_MAX && K <= INT32_MAX,
              "k80_gguf: weight too large");
  return K;
}

}  // namespace k80_gguf

// y [M, N] = x [M, K] @ W^T for a GGUF Q4_K or Q6_K weight, for any M
// (launched 8 rows at a time).
torch::Tensor k80_gguf_gemm(torch::Tensor const& x, torch::Tensor const& qweight,
                            int64_t type) {
  using namespace k80_gguf;
  const int64_t K = check_weight(qweight, type);
  const int N = qweight.size(0);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K && x.is_contiguous() &&
                  x.device() == qweight.device(),
              "k80_gguf_gemm: x must be contiguous [M, K] on the weight's device");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "k80_gguf_gemm: x must be float32 or float16");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const int M = x.size(0);
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto f32 = x.options().dtype(at::kFloat);
  auto ys = torch::empty({M, K}, f32);
  auto ysum16 = torch::empty({type == kTypeQ4K ? M : 0, K / 16}, f32);
  float* ysum16_p = type == kTypeQ4K ? ysum16.data_ptr<float>() : nullptr;
  const dim3 pgrid(K / 256, M);
  if (x.scalar_type() == at::kFloat)
    prep_kernel<float><<<pgrid, 256, 0, stream>>>(
        x.data_ptr<float>(), ys.data_ptr<float>(), ysum16_p, K);
  else
    prep_kernel<c10::Half><<<pgrid, 256, 0, stream>>>(
        x.data_ptr<c10::Half>(), ys.data_ptr<float>(), ysum16_p, K);
  const uint8_t* w = qweight.data_ptr<uint8_t>();
  const size_t row_bytes = qweight.size(1);
  for (int m0 = 0; m0 < M; m0 += kMaxRows) {
    const int m = std::min(kMaxRows, M - m0);
    const float* ys_p = ys.data_ptr<float>() + (size_t)m0 * K;
    const float* s_p = ysum16_p ? ysum16_p + (size_t)m0 * (K / 16) : nullptr;
    if (x.scalar_type() == at::kFloat)
      gemm_rows<float>(m, type, w, ys_p, s_p,
                       y.data_ptr<float>() + (size_t)m0 * N, N, K, row_bytes,
                       stream);
    else
      gemm_rows<c10::Half>(m, type, w, ys_p, s_p,
                           y.data_ptr<c10::Half>() + (size_t)m0 * N, N, K,
                           row_bytes, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// out [N, K] (float32 or float16) = the dequantized GGUF Q4_K or Q6_K weight.
void k80_gguf_dequantize(torch::Tensor& out, torch::Tensor const& qweight,
                         int64_t type) {
  using namespace k80_gguf;
  const int64_t K = check_weight(qweight, type);
  const int64_t N = qweight.size(0);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == N && out.size(1) == K &&
                  out.is_contiguous() && out.device() == qweight.device(),
              "k80_gguf_dequantize: out must be contiguous [N, K]");
  TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kHalf,
              "k80_gguf_dequantize: out must be float32 or float16");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const int64_t nblocks = N * (K / kQK);
  if (nblocks == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* w = qweight.data_ptr<uint8_t>();
  // 256 threads: 8 Q4_K blocks (a warp each) or 16 Q6_K blocks (16 lanes each)
  const int per = type == kTypeQ4K ? 8 : 16;
  const dim3 grid((nblocks + per - 1) / per);
  if (type == kTypeQ4K) {
    if (out.scalar_type() == at::kFloat)
      q4k_dequant_kernel<float><<<grid, 256, 0, stream>>>(w, out.data_ptr<float>(), nblocks);
    else
      q4k_dequant_kernel<c10::Half>
          <<<grid, 256, 0, stream>>>(w, out.data_ptr<c10::Half>(), nblocks);
  } else {
    if (out.scalar_type() == at::kFloat)
      q6k_dequant_kernel<float><<<grid, 256, 0, stream>>>(w, out.data_ptr<float>(), nblocks);
    else
      q6k_dequant_kernel<c10::Half>
          <<<grid, 256, 0, stream>>>(w, out.data_ptr<c10::Half>(), nblocks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
