#include "CudaSolver.h"
#include <cmath>
#include "cublas_v2.h"
#include "cuda_runtime.h"

CudaSolver::CudaSolver(const int n) : n(n) {

}

CudaSolver::~CudaSolver() {

}

void CudaSolver::solve(float *A, float *b, int maxIterations, float tolerance, float *res) {
    //copy A, b to gpu

    float error = MAXFLOAT;
    while (error >= tolerance) {
        error = iterate();
    }

    cudaMemcpy(res, x_k, sizeof(float) * n, cudaMemcpyDeviceToHost);
    //transfer x_k back
}

float CudaSolver::iterate() {
    return 0;
}
