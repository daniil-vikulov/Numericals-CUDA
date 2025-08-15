#ifndef ITERATIVEMETHOD_HPP
#define ITERATIVEMETHOD_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

// iterative SLAE solution
int iterativeMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
                    std::vector<double>& x, int max_iters, double delta);

void sparseMatrixVectorMultiply(const std::vector<double>& values,
                                const std::vector<int>& row_indices,
                                const std::vector<int>& col_ptrs, const std::vector<double>& x,
                                std::vector<double>& result, int N);

int iterativeMethodSparse(const std::vector<double>& values, const std::vector<int>& row_indices,
                          const std::vector<int>& col_ptrs, const std::vector<double>& b,
                          std::vector<double>& x, int max_iters, double delta, int N);

#endif  // ITERATIVEMETHOD_HPP
