// SPDX-License-Identifier: BSD-3-Clause
//
// Minimal CUTLASS SGEMM reproducer that dispatches through cutlass::arch::Sm37.
//
// Proves that:
//   1. NVCC compiles a CUTLASS GEMM with TORCH_CUDA_ARCH_LIST=3.7
//      against patched CUTLASS (sm37-trait.patch from #25 applied).
//   2. The compiled binary executes correctly on a Tesla K80 (sm_37 / Kepler).
//
// FP32 only — Kepler has no tensor cores, no native FP16 compute.
// SIMT path only — no tensor-op gating reachable.
//
// Build via the build.sh in this directory, or directly:
//   nvcc -std=c++17 -arch=sm_37 \
//        -I /path/to/patched/cutlass/include \
//        -I /path/to/patched/cutlass/tools/util/include \
//        sgemm_sm37.cu -o sgemm_sm37
//
// Run with no args; exits 0 on success, non-zero on numerical error.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(_err) << std::endl; \
        return 2; \
    } \
} while (0)

int main() {
    // Use the Sm37 arch tag added by sm37-trait.patch.
    // OpClassSimt uses scalar FP32 FMA only, which sm_37 (Kepler GK210)
    // supports natively.
    using ArchTag = cutlass::arch::Sm37;
    using OpClass = cutlass::arch::OpClassSimt;

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        float, ColumnMajor,    // A
        float, ColumnMajor,    // B
        float, ColumnMajor,    // C
        float,                 // accumulator
        OpClass,
        ArchTag                // <-- the new sm_37 tag
    >;

    // Problem size: small enough to fit comfortably on K80 even with margin
    // for other workloads. Square 256x256x256 = 64 KB per matrix, ~256 KB total.
    int constexpr M = 256;
    int constexpr N = 256;
    int constexpr K = 256;

    // Deterministic inputs: A and B are all-ones, so each output element of
    // D = alpha * A * B + beta * C should equal alpha * K (with beta = 0).
    // alpha = 1.0 -> expected = K = 256.
    int constexpr lda = M;  // column-major: leading dim = num rows
    int constexpr ldb = K;
    int constexpr ldc = M;

    std::vector<float> A_h(M * K, 1.0f);
    std::vector<float> B_h(K * N, 1.0f);
    std::vector<float> D_h(M * N, 0.0f);

    float* A_d = nullptr;
    float* B_d = nullptr;
    float* D_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_d, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(),
                          M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(),
                          K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(D_d, 0, M * N * sizeof(float)));

    Gemm gemm_op;

    Gemm::Arguments args(
        {M, N, K},
        {A_d, lda},
        {B_d, ldb},
        {D_d, ldc},     // C = D, no separate input C
        {D_d, ldc},
        {1.0f, 0.0f}    // alpha, beta
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed: "
                  << cutlassGetStatusString(status) << std::endl;
        return 3;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(D_h.data(), D_d,
                          M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(D_d);

    // Each output element should equal K = 256.0f.
    float constexpr expected = static_cast<float>(K);
    float constexpr tolerance = 1e-3f;
    int errors = 0;
    float max_abs_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(D_h[i] - expected);
        max_abs_error = std::max(max_abs_error, err);
        if (err > tolerance) {
            ++errors;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "=== CUTLASS sm_37 SGEMM reproducer ===\n";
    std::cout << "device       : " << prop.name << " (cc "
              << prop.major << "." << prop.minor << ")\n";
    std::cout << "problem      : " << M << " x " << N << " x " << K
              << " (M x N x K), fp32, SIMT\n";
    std::cout << "expected     : " << expected << " per element\n";
    std::cout << "got D[0]     : " << D_h[0] << "\n";
    std::cout << "max abs err  : " << std::scientific << max_abs_error << "\n";
    std::cout << "tolerance    : " << tolerance << "\n";
    std::cout << "errors       : " << errors << " / " << (M * N) << "\n";

    if (errors == 0) {
        std::cout << "RESULT       : PASS\n";
        return 0;
    } else {
        std::cout << "RESULT       : FAIL\n";
        return 1;
    }
}
