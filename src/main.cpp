#include "CudaSolver.h"
#include <iostream>
#include <cassert>

#define N 3

float* unwrap(float A[N][N]) {
    const auto A_flat = new float[N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_flat[i * N + j] = A[i][j];
        }
    }

    return A_flat;
}

float* multiply(const float* A, const float* x, int n) {
    // Allocate memory for the result vector Ax
    auto Ax = new float[n];

    // Perform the matrix-vector multiplication
    for (int i = 0; i < n; ++i) {
        Ax[i] = 0.0f;
        for (int j = 0; j < n; ++j) {
            Ax[i] += A[i * n + j] * x[j];
        }
    }

    return Ax;
}

int main () {
    CudaSolver solver(N);

    float A[N][N] = {{4, 1, 1}, {1, 3, 0}, {1, 0, 2}};
    float b[N] = {10, -8, 12};

    float res[N] = {};

    const auto A_1D = unwrap(A);

    assert(solver.solve(A_1D, b, 100, 1e-7, res));
    multiply(A_1D, res, N);

    for (const auto el : res){
        std::cout << el << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < N; ++i) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;

    delete[] A_1D;
    //delete[] product;

    return 0;
}