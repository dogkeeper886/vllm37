// SPDX-License-Identifier: BSD-3-Clause

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
    using ArchTag = cutlass::arch::Sm37;
    using OpClass = cutlass::arch::OpClassSimt;

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        float, ColumnMajor,
        float, ColumnMajor,
        float, ColumnMajor,
        float,
        OpClass,
        ArchTag
    >;

    // M, N and K: whole numbers from 1
    int constexpr M = 256;
    int constexpr N = 256;
    int constexpr K = 256;

    int constexpr lda = M;
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
        {D_d, ldc},
        {D_d, ldc},
        {1.0f, 0.0f}
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

    float constexpr expected = static_cast<float>(K);
    // an absolute error above 0
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
