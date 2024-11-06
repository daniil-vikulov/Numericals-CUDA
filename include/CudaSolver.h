#pragma once

class CudaSolver {
private:
    float *x_k, r_k, r_k_prev, p_k, b;
    int n;

public:
    ///@brief instantiates CUDA tool, which finds a vector X in the following equation: AX = B, where A is a known
    ///matrix nxn and B is a known vector of n elements
    explicit CudaSolver(int n);

    ~CudaSolver();

    void solve(float *A, float *b, int maxIterations, float tolerance, float *res);

private:
    float iterate();
};