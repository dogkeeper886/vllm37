// SPDX-License-Identifier: BSD-3-Clause
//
// Numerical correctness check.
// Runs the same SGEMM through both CUTLASS (with cutlass::arch::Sm37, SIMT
// path) and cuBLAS, compares element-wise, and reports tolerance achieved
// across multiple matrix sizes with randomized inputs.
//
// Bit-exact agreement is unlikely: the two implementations sum K
// floating-point products in different orderings (different tile schedules,
// different warp partitioning), and FP32 addition is non-associative. We
// expect O(K * eps * |A| * |B|) absolute error in the accumulator, which for
// our random inputs in [-1, 1] and K up to 1024 means ~1e-4 absolute error
// and ~1e-6 relative error.
//
// Acceptance: max relative error < 1e-3 across all tested sizes — comfortably
// inside float precision noise; tighter than fp16 inference tolerance budgets.
//
// Build: see build.sh (needs -lcublas added to the link line vs sgemm_sm37.cu).

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(_err) << std::endl; \
        std::exit(2); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                  << ": status=" << _s << std::endl; \
        std::exit(2); \
    } \
} while (0)

// CUTLASS SGEMM via the arch::Sm37 tag (cutlass-patches/sm37-trait.patch).
// Column-major to match cuBLAS conventions.
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm37
>;

struct ErrorStats {
    float max_abs;     // max element-wise absolute error
    float max_rel;     // max relative error (where denom is reasonable)
    int max_abs_idx;   // location of max abs error
    double mean_abs;
};

ErrorStats compare(const std::vector<float>& a, const std::vector<float>& b) {
    ErrorStats s{0.0f, 0.0f, 0, 0.0};
    double sum_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        sum_abs += diff;
        if (diff > s.max_abs) {
            s.max_abs = diff;
            s.max_abs_idx = static_cast<int>(i);
        }
        // Avoid divide-by-tiny: only count rel error where the magnitude is
        // big enough that abs error of float-precision noise wouldn't dominate.
        float denom = std::max(std::abs(a[i]), std::abs(b[i]));
        if (denom > 1e-3f) {
            float rel = diff / denom;
            if (rel > s.max_rel) s.max_rel = rel;
        }
    }
    s.mean_abs = sum_abs / static_cast<double>(a.size());
    return s;
}

// Run one (M,N,K) test case and return whether it met tolerance.
bool run_case(int M, int N, int K, std::mt19937& rng, float rel_tolerance) {
    std::cout << "\n--- M=" << M << " N=" << N << " K=" << K << " ---\n";

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_h(M * K);
    std::vector<float> B_h(K * N);
    for (auto& v : A_h) v = dist(rng);
    for (auto& v : B_h) v = dist(rng);

    int const lda = M;  // column-major
    int const ldb = K;
    int const ldc = M;

    float *A_d, *B_d, *C_cutlass_d, *C_cublas_d;
    CUDA_CHECK(cudaMalloc(&A_d, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_cutlass_d, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_cublas_d, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C_cutlass_d, 0, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(C_cublas_d, 0, M * N * sizeof(float)));

    // CUTLASS path.
    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {M, N, K},
        {A_d, lda},
        {B_d, ldb},
        {C_cutlass_d, ldc},
        {C_cutlass_d, ldc},
        {1.0f, 0.0f}
    );
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "  CUTLASS GEMM failed: "
                  << cutlassGetStatusString(status) << std::endl;
        cudaFree(A_d); cudaFree(B_d);
        cudaFree(C_cutlass_d); cudaFree(C_cublas_d);
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLAS reference path.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             A_d, lda,
                             B_d, ldb,
                             &beta,
                             C_cublas_d, ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDestroy(handle));

    // Pull both results back, compare.
    std::vector<float> C_cutlass_h(M * N), C_cublas_h(M * N);
    CUDA_CHECK(cudaMemcpy(C_cutlass_h.data(), C_cutlass_d,
                          M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C_cublas_h.data(), C_cublas_d,
                          M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(A_d); cudaFree(B_d);
    cudaFree(C_cutlass_d); cudaFree(C_cublas_d);

    ErrorStats s = compare(C_cutlass_h, C_cublas_h);
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  max abs error : " << s.max_abs
              << " (at element " << s.max_abs_idx << ")\n";
    std::cout << "  max rel error : " << s.max_rel
              << " (tolerance " << rel_tolerance << ")\n";
    std::cout << "  mean abs error: " << s.mean_abs << "\n";

    bool pass = s.max_rel <= rel_tolerance;
    std::cout << "  result        : " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "=== CUTLASS Sm37 vs cuBLAS SGEMM ===\n";
    std::cout << "device       : " << prop.name << " (cc "
              << prop.major << "." << prop.minor << ")\n";

    // Fixed seed → deterministic, reproducible across CI runs.
    std::mt19937 rng(20260426u);

    // Tolerance budget: O(K * eps) = O(K * 2^-23) ≈ 1.2e-4 for K=1024.
    // Pick 1e-3 as the headline tolerance — comfortably inside what fp32
    // SGEMM should produce regardless of reduction order, and tighter than
    // fp16 inference tolerance budgets (1e-2 typical).
    float const rel_tolerance = 1e-3f;

    int const sizes[][3] = {
        {64,   64,   64},
        {128,  128,  128},
        {256,  256,  256},
        {512,  512,  512},
        {1024, 1024, 1024},
    };

    int total = 0, passed = 0;
    for (auto const& s : sizes) {
        ++total;
        if (run_case(s[0], s[1], s[2], rng, rel_tolerance)) ++passed;
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "passed: " << passed << "/" << total << "\n";
    std::cout << "RESULT       : " << (passed == total ? "PASS" : "FAIL") << "\n";
    return passed == total ? 0 : 1;
}
