#pragma once

#include <cublas_v2.h>

class CudaSolver {
    const int n{};
    cublasHandle_t handle{};
    int k{};
    float *A{};
    float *x_k{};
    float *r_k{};
    float *p_k{};
    float *b{};
    float *r_k_new{};
    float *tmp{};
    static constexpr float one = 1.0f;
    static constexpr float zero = 0.0f;

public:
    ///@brief instantiates CUDA tool, which finds a vector X in the following equation: AX = B, where A is a known
    ///matrix nxn and B is a known vector of n elements
    explicit CudaSolver(int n);

    ~CudaSolver();

    bool solve(const float *A, const float *b, int maxIterations, float tolerance, float *res);

private:
    [[nodiscard]] float iterate() const;

    [[nodiscard]] inline float calc_alpha() const;

    inline void update_x(float alpha) const;

    inline void update_r_k_new(float alpha) const;

    [[nodiscard]] inline float calc_beta() const;

    inline void update_p(float beta) const;

    inline float dot(const float *v1, const float *v2) const;
};
