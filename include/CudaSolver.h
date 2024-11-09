#pragma once

#include <cublas_v2.h>
#include "CuBLASTemplate.h"
#include <iostream>

#define CB(call) { \
cublasStatus_t status = (call); \
if (status != CUBLAS_STATUS_SUCCESS) { \
throw std::runtime_error("cuBLAS Error at line " + std::to_string(__LINE__) + ": " + std::to_string(status)); \
} \
}

#define CC(call) { \
cudaError_t status = (call); \
if (status != cudaSuccess) { \
throw std::runtime_error("CUDA Error at line " + std::to_string(__LINE__) + ": " + cudaGetErrorString(status)); \
} \
}

template <typename T, int N>
class CudaSolver {
    constexpr static int n = N;
    cublasHandle_t handle{};
    int k{};
    T *A{};
    T *x_k{};
    T *r_k{};
    T *p_k{};
    T *b{};
    T *r_k_new{};
    T *tmp{};
    static constexpr T one = static_cast<T>(1.0);
    static constexpr T zero = static_cast<T>(0.0);

public:
    /// @brief instantiates CUDA tool, which finds a vector X in the following equation: AX = B, where A is a known
    /// matrix nxn and B is a known vector of n elements
    explicit CudaSolver();

    ~CudaSolver();

    bool solve(const T *A, const T *b, int maxIterations, T tolerance, T *res);

private:
    [[nodiscard]] T iterate() const;

    [[nodiscard]] T calc_alpha() const;

    void update_x(T alpha) const;

    void update_r_k_new(T alpha) const;

    [[nodiscard]] T calc_beta() const;

    void update_p(T beta) const;

    T dot(const T *v1, const T *v2) const;
};

template <typename T, int N>
CudaSolver<T, N>::CudaSolver() {
    // Init
    CB(cublasCreate(&handle));

    CC(cudaMalloc(reinterpret_cast<void **>(&A), n * n * sizeof(T)));
    CC(cudaMalloc(reinterpret_cast<void **>(&b), n * sizeof(T)));
    CC(cudaMalloc(reinterpret_cast<void **>(&x_k), n * sizeof(T)));
    CC(cudaMalloc(reinterpret_cast<void **>(&r_k), n * sizeof(T)));
    CC(cudaMalloc(reinterpret_cast<void **>(&r_k_new), n * sizeof(T)));
    CC(cudaMalloc(reinterpret_cast<void **>(&p_k), n * sizeof(T)));
    CC(cudaMalloc(reinterpret_cast<void **>(&tmp), n * sizeof(T)));
}

template <typename T, int N>
CudaSolver<T, N>::~CudaSolver() {
    try {
        // Free
        CC(cudaFree(A));
        CC(cudaFree(b));
        CC(cudaFree(x_k));
        CC(cudaFree(r_k));
        CC(cudaFree(r_k_new));
        CC(cudaFree(p_k));
        CC(cudaFree(tmp));

        CB(cublasDestroy(handle));
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

template <typename T, int N>
bool CudaSolver<T, N>::solve(const T *cpu_A, const T *cpu_b, const int maxIterations, const T tolerance, T *res) {
    k = 0;

    CC(cudaMemcpy(A, cpu_A, n * n * sizeof(T), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(b, cpu_b, n * sizeof(T), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(r_k, b, n * sizeof(T), cudaMemcpyDeviceToDevice));
    CC(cudaMemcpy(p_k, r_k, n * sizeof(T), cudaMemcpyDeviceToDevice));
    CC(cudaMemset(x_k, 0, n * sizeof(T)));

    T error = std::numeric_limits<T>::max();
    while (error >= tolerance && k < maxIterations) {
        error = iterate();
        std::cout << "Error" << error << std::endl;
        k++;
    }

    CC(cudaMemcpy(res, x_k, sizeof(T) * n, cudaMemcpyDeviceToHost));

    if (k >= maxIterations) {
        return false;
    }

    std::cout << "Total iterations: " << k << std::endl;
    return true;
}

template <typename T, int N>
T CudaSolver<T, N>::iterate() const {
    const T alpha = calc_alpha();

    update_x(alpha);
    update_r_k_new(alpha);

    const T beta = calc_beta();
    update_p(beta);

    CC(cudaMemcpy(r_k, r_k_new, n * sizeof(T), cudaMemcpyDeviceToDevice));

    return std::sqrt(dot(r_k, r_k));
}

template <typename T, int N>
T CudaSolver<T, N>::calc_alpha() const {
    // return <r_k, r_k> / <p_k, A * p_k>

    // tmp = A * p_k
    CB(cublasGemv(handle, CUBLAS_OP_N, n, n, &one, A, n, p_k, 1, &zero, tmp, 1));

    // <p_k, A*p_k>
    T denom_result = dot(p_k, tmp);

    return dot(r_k, r_k) / denom_result;
}

template <typename T, int N>
void CudaSolver<T, N>::update_x(const T alpha) const {
    // x_k = x_k + alpha * p_k

    CB(cublasAxpy(handle, n, &alpha, p_k, 1, x_k, 1));
}

template <typename T, int N>
void CudaSolver<T, N>::update_r_k_new(const T alpha) const {
    // r_k_new = r_k - alpha * A * p_k

    const T neg_alpha = -alpha;
    // tmp = -alpha * A * p_k
    CB(cublasGemv(handle, CUBLAS_OP_N, n, n, &neg_alpha, A, n, p_k, 1, &zero, tmp, 1));

    // r_k_new = r_k
    CC(cudaMemcpy(r_k_new, r_k, n * sizeof(T), cudaMemcpyDeviceToDevice));

    //r_k_new = tmp + r_k_new
    CB(cublasAxpy(handle, n, &one, tmp, 1, r_k_new, 1));
}

template <typename T, int N>
T CudaSolver<T, N>::calc_beta() const {
    // return <r_{k+1}, r_{k+1}> / <r_k, r_k>

    return 0;
    //return dot(r_k_new, r_k_new) / dot(r_k, r_k);
}

template <typename T, int N>
void CudaSolver<T, N>::update_p(const T beta) const {
    // p_k = r_k_new + beta * p_k

    // p_K = beta * p_k
    CB(cublasScal(handle, n, &beta, p_k, 1));

    // p_k = r_k_new + p_K
    CB(cublasAxpy(handle, n, &one, r_k_new, 1, p_k, 1));
}

template <typename T, int N>
T CudaSolver<T, N>::dot(const T *v1, const T *v2) const {
    // return <v1, v2>

    T result = 0;
    CB(cublasDot(handle, n, v1, 1, v2, 1, &result));
    return result;
}