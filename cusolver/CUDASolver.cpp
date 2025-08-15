#include "CUDASolver.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>

CUDASolver::CUDASolver(const std::vector<std::vector<double> > &a, int p,
                       const std::vector<double> &b) : _res(b.size()), _size(static_cast<int>(b.size())), _power(p) {
    assert(a.size() == b.size());

    cudaMalloc(&_a, _size * _size * sizeof(double));
    cudaMalloc(&_a_p, _size * _size * sizeof(double));
    cudaMalloc(&_a_temp, _size * _size * sizeof(double));
    cudaMalloc(&_b, _size * sizeof(double));
    cudaMalloc(&_x, _size * sizeof(double));

    for (int i = 0; i < _size; ++i) {
        cudaMemcpy(&_a[i * _size], a[i].data(), _size * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(_a_p, _a, _size * _size * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(_b, b.data(), _size * sizeof(double), cudaMemcpyHostToDevice);

    cublasCreate(&_blasHandle);
    cusolverDnCreate(&_solverHandle);
}

CUDASolver::~CUDASolver() {
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_x);
    cudaFree(_a_p);
    cudaFree(_a_temp);

    cublasDestroy(_blasHandle);
    cusolverDnDestroy(_solverHandle);
}

std::vector<double> CUDASolver::solve() {
    matrixPower();
    solveSystem();
    cudaMemcpy(_res.data(), _x, _size * sizeof(double), cudaMemcpyDeviceToHost);

    return std::move(_res);
}

void CUDASolver::multiplyMatrix() const {
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    cublasDgemm(_blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                _size, _size, _size,
                &alpha, _a_p, _size, _a, _size,
                &beta, _a_temp, _size);

    cudaMemcpy(_a_p, _a_temp, _size * _size * sizeof(double), cudaMemcpyDeviceToDevice);
}

void CUDASolver::solveSystem() const {
    int *d_devIpiv, *d_info;
    cudaMalloc(&d_devIpiv, _size * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    int work_size = 0;
    double *d_work;
    cusolverDnDgetrf_bufferSize(_solverHandle, _size, _size, _a_p, _size, &work_size);
    cudaMalloc(&d_work, work_size * sizeof(double));

    cusolverDnDgetrf(_solverHandle, _size, _size, _a_p, _size, d_work, d_devIpiv, d_info);

    cusolverDnDgetrs(_solverHandle, CUBLAS_OP_T, _size, 1, _a_p, _size, d_devIpiv, _b, _size, d_info);

    cudaMemcpy(_x, _b, _size * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaFree(d_devIpiv);
    cudaFree(d_info);
    cudaFree(d_work);
}

void CUDASolver::matrixPower() const {
    for (int i = 1; i < _power; ++i) {
        multiplyMatrix();
    }
}
