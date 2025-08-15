#include "iterative_method.hpp"

int iterativeMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
                    std::vector<double>& x, int max_iters, double delta) {
    int N = A.size();
    std::vector<double> x_old(N, 0.0);
    std::vector<double> x_new(N);

    for (int k = 0; k < max_iters; ++k) {
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j != i) {
                    sum += A[i][j] * x_old[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        double norm = 0.0;
        for (int i = 0; i < N; ++i) {
            norm += std::pow(x_new[i] - x_old[i], 2);
        }
        norm = std::sqrt(norm);

        if (norm < delta) {
            x = x_new;
            return 0;  // converges
        }

        x_old = x_new;
    }
    return 1;  // diverges
}

// for sparse matrix
void sparseMatrixVectorMultiply(const std::vector<double>& values,
                                const std::vector<int>& row_indices,
                                const std::vector<int>& col_ptrs, const std::vector<double>& x,
                                std::vector<double>& result, int N) {
    for (int i = 0; i < N; ++i) {
        result[i] = 0.0;
        for (int j = col_ptrs[i]; j < col_ptrs[i + 1]; ++j) {
            result[i] += values[j] * x[row_indices[j]];
        }
    }
}

int iterativeMethodSparse(const std::vector<double>& values, const std::vector<int>& row_indices,
                          const std::vector<int>& col_ptrs, const std::vector<double>& b,
                          std::vector<double>& x, int max_iters, double delta, int N) {
    std::vector<double> x_old(N, 0.0);
    std::vector<double> x_new(N);
    int iter = 0;

    while (iter < max_iters) {
        sparseMatrixVectorMultiply(values, row_indices, col_ptrs, x_old, x_new, N);

        for (int i = 0; i < N; ++i) {
            x_new[i] = (b[i] - x_new[i]) / values[col_ptrs[i]];
        }

        double norm = 0.0;
        for (int i = 0; i < N; ++i) {
            norm += (x_new[i] - x_old[i]) * (x_new[i] - x_old[i]);
        }
        norm = std::sqrt(norm);
        if (norm < delta) {
            x = x_new;
            return 0;
        }

        x_old = x_new;
        ++iter;
    }

    return 1;
}