#pragma once

#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>

class CUDASolver {
    double *_a{};
    double *_a_p{};
    double *_a_temp{};
    double *_b{};
    double *_x{};

    std::vector<double> _res;

    int _size;
    int _power;
    cublasHandle_t _blasHandle{};
    cusolverDnHandle_t _solverHandle{};

public:
    /// Solves a^p * x = b.
    CUDASolver(const std::vector<std::vector<double> > &a, int p,
               const std::vector<double> &b);

    ~CUDASolver();

    /// Solves equasion. Returns result vector x
    std::vector<double> solve();

private:
    void matrixPower() const;

    void solveSystem() const;

    void multiplyMatrix() const;
};
