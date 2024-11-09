#include "CudaSolver.h"
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CC(call) { \
cudaError_t status = (call); \
if (status != cudaSuccess) { \
throw std::runtime_error("CUDA Error at line " + std::to_string(__LINE__) + ": " + cudaGetErrorString(status)); \
} \
}

// Macro for checking cuBLAS errors
#define CB(call) { \
cublasStatus_t status = (call); \
if (status != CUBLAS_STATUS_SUCCESS) { \
throw std::runtime_error("cuBLAS Error at line " + std::to_string(__LINE__) + ": " + std::to_string(status)); \
} \
}

void printCudaVector(float* d_v, int n) {
    // Allocate host memory to copy the device data
    float* h_v = new float[n];

    // Copy data from device memory to host memory
    cudaMemcpy(h_v, d_v, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the vector elements
    for (int i = 0; i < n; ++i) {
        std::cout << h_v[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] h_v;
}

CudaSolver::CudaSolver(const int n) : n(n) {
    // Init
    CB(cublasCreate(&handle));

    CC(cudaMalloc(reinterpret_cast<void **>(&A), n * n * sizeof(float)));
    CC(cudaMalloc(reinterpret_cast<void **>(&b), n * sizeof(float)));
    CC(cudaMalloc(reinterpret_cast<void **>(&x_k), n * sizeof(float)));
    CC(cudaMalloc(reinterpret_cast<void **>(&r_k), n * sizeof(float)));
    CC(cudaMalloc(reinterpret_cast<void **>(&r_k_new), n * sizeof(float)));
    CC(cudaMalloc(reinterpret_cast<void **>(&p_k), n * sizeof(float)));
    CC(cudaMalloc(reinterpret_cast<void **>(&tmp), n * sizeof(float)));
}

CudaSolver::~CudaSolver() {
    // Free

    CC(cudaFree(A));
    CC(cudaFree(b));
    CC(cudaFree(x_k));
    CC(cudaFree(r_k));
    CC(cudaFree(r_k_new));
    CC(cudaFree(p_k));
    CC(cudaFree(tmp));

    CB(cublasDestroy(handle));
}

bool CudaSolver::solve(const float *cpu_A, const float *cpu_b, const int maxIterations, const float tolerance,
                       float *res) {
    k = 0;

    CC(cudaMemcpy(A, cpu_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(b, cpu_b, n * sizeof(float), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(r_k, b, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CC(cudaMemcpy(p_k, r_k, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CC(cudaMemset(x_k, 0, n));


    float error = MAXFLOAT;
    while (error >= tolerance && k < maxIterations) {
        error = iterate();
        k++;
    }

    if (k >= maxIterations) {
        return false;
    }

    // ReSharper disable once CppNoDiscardExpression
    iterate();

    CC(cudaMemcpy(res, x_k, sizeof(float) * n, cudaMemcpyDeviceToHost));

    std::cout << "Total iterations: " << k << std::endl;

    return true;
}

float CudaSolver::iterate() const {
    const float alpha = calc_alpha();

    update_x(alpha);
    update_r_k_new(alpha);

    const float beta = calc_beta();

    update_p(beta);

    CC(cudaMemcpy(r_k, r_k_new, n * sizeof(float), cudaMemcpyDeviceToDevice));

    //printCudaVector(r_k, n);
    return dot(r_k, r_k);
}

float CudaSolver::calc_alpha() const {
    // return <r_k, r_k> / <p_k, A * p_k>

    // tmp = A * p_k
    CB(cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, A, n, p_k, 1, &zero, tmp, 1));

    // <p_k, A*p_k>
    float denom_result = dot(p_k, tmp);

    printCudaVector(r_k, n);
    std::cout << dot(r_k, r_k) / denom_result << std::endl;

    return dot(r_k, r_k) / denom_result;
}

void CudaSolver::update_x(const float alpha) const {
    // x_k = x_k + alpha * p_k
    CB(cublasSaxpy(handle, n, &alpha, p_k, 1, x_k, 1));
}

void CudaSolver::update_r_k_new(const float alpha) const {
    // r_k_new = r_k - alpha * A * p_k

    const float neg_alpha = -alpha;

    // tmp = -alpha * A * p_k
    CB(cublasSgemv(handle, CUBLAS_OP_N, n, n, &neg_alpha, A, n, p_k, 1, &zero, tmp, 1));

    // r_k_new = r_k
    CC(cudaMemcpy(r_k_new, r_k, n * sizeof(float), cudaMemcpyDeviceToDevice));

    //r_k_new = tmp + r_k_new
    CB(cublasSaxpy(handle, n, &one, tmp, 1, r_k_new, 1));
}

float CudaSolver::calc_beta() const {
    // return <r_{k+1}, r_{k+1}> / <r_k, r_k>


    std::cout << "Beta components:" << dot(r_k_new, r_k_new) << " " << dot(r_k, r_k) << std::endl;
    return dot(r_k_new, r_k_new) / dot(r_k, r_k);
}

void CudaSolver::update_p(const float beta) const {
    // p_k = r_k_new + beta * p_k

    // p_K = beta * p_k
    CB(cublasSscal(handle, n, &beta, p_k, 1));

    // p_k = r_k_new + p_K
    CB(cublasSaxpy(handle, n, &one, r_k_new, 1, p_k, 1));
}

float CudaSolver::dot(const float *v1, const float *v2) const {
    // return <v1, v2>

    float result = 0;
    CB(cublasSdot(handle, n, v1, 1, v2, 1, &result));

    return result;
}
