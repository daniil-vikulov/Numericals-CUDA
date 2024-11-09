#include "CudaSolver.h"
#include <iostream>
#include <cassert>
#include <iomanip>

#define N 5
#define DATA_TYPE float

template<typename T>
T* unwrap(T A[N][N]) {
    const auto A_flat = new T[N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_flat[i * N + j] = A[i][j];
        }
    }

    return A_flat;
}

template<typename T>
T* multiply(const T* A, const T* x, const int n) {
    auto Ax = new T[n];

    for (int i = 0; i < n; ++i) {
        Ax[i] = 0.0f;
        for (int j = 0; j < n; ++j) {
            Ax[i] += A[i * n + j] * x[j];
        }
    }

    return Ax;
}

int main () {
    CudaSolver<DATA_TYPE, N> solver;

    DATA_TYPE A[N][N] = {
        {4.0, 1.0, 0.0, 0.0, 0.0},
        {1.0, 3.0, 1.0, 0.0, 0.0},
        {0.0, 1.0, 2.0, 1.0, 0.0},
        {0.0, 0.0, 1.0, 3.0, 1.0},
        {0.0, 0.0, 0.0, 1.0, 4.0}
    };

    DATA_TYPE b[N] = {6.0, 11.0, 13.0, 19.0, 21.0};

    DATA_TYPE res[N] = {};

    const auto A_1D = unwrap<DATA_TYPE>(A);

    if (!solver.solve(A_1D, b, 50, 1e-15, res)) {
        std::cout << "Reached max iterations!" << std::endl;
    }
    multiply(A_1D, res, N);

    for (const auto el : res) {
        std::cout << std::fixed << std::setprecision(10) << el << " ";
    }
    std::cout << std::endl;
    for (auto el : res) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    delete[] A_1D;

    return 0;
}