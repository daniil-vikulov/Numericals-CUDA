#include "gauss.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <Accelerate/Accelerate.h>

void convert_to_column_major(const std::vector<std::vector<double>> &A, double *A_col_major,
                             int N) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            A_col_major[c * N + r] = A[r][c];
        }
    }
}

static int arg_max(const std::vector<std::vector<double>> &A, const std::vector<int> &row_order,
                   int start_row, int col) {
    double max_val = std::abs(A[row_order[start_row]][col]);
    int max_row = start_row;
    for (int i = start_row + 1; i < row_order.size(); ++i) {
        if (std::abs(A[row_order[i]][col]) >= max_val) {
            max_val = std::abs(A[row_order[i]][col]);
            max_row = i;
        }
    }
    return max_row;
}

void gauss::solve_slae_gauss(std::vector<std::vector<double>> A, std::vector<double> b,
                             std::vector<double> &x) {
    int n = A.size();
    x.resize(n, 0.0);

    std::vector<int> row_order(n), col_order(n);
    for (int i = 0; i < n; ++i)
        row_order[i] = i;
    for (int j = 0; j < n; ++j)
        col_order[j] = j;

    for (int row = 0; row < n; ++row) {
        int max_row = arg_max(A, row_order, row, col_order[row]);
        std::swap(row_order[row], row_order[max_row]);

        int max_col = row;
        for (int col = row + 1; col < n; ++col) {
            if (std::abs(A[row_order[row]][col_order[col]]) >=
                std::abs(A[row_order[row]][col_order[max_col]])) {
                max_col = col;
            }
        }
        std::swap(col_order[row], col_order[max_col]);

        double pivot_value = A[row_order[row]][col_order[row]];
        for (int k = row + 1; k < n; ++k) {
            double factor = A[row_order[k]][col_order[row]] / pivot_value;
            b[row_order[k]] -= factor * b[row_order[row]];
            for (int j = row; j < n; ++j) {
                A[row_order[k]][col_order[j]] -= factor * A[row_order[row]][col_order[j]];
            }
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        double sum = b[row_order[i]];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[row_order[i]][col_order[j]] * x[col_order[j]];
        }
        x[col_order[i]] = sum / A[row_order[i]][col_order[i]];
    }
}

void gauss::solve_slae_lapack(std::vector<std::vector<double>> &A, std::vector<double> &b,
                              std::vector<double> &x) {
    int N = A.size();

    double *A_col_major = new double[N * N];
    convert_to_column_major(A, A_col_major, N);

    double *b_array = new double[N];
    std::copy(b.begin(), b.end(), b_array);

    int *ipiv = new int[N];
    int info;
    int nrhs = 1;

    dgesv_(&N, &nrhs, A_col_major, &N, ipiv, b_array, &N, &info);

    if (info == 0) {
        x.assign(b_array, b_array + N);
    } else {
        std::cerr << "LAPACK dgesv failed with error code: " << info << std::endl;
    }

    delete[] A_col_major;
    delete[] b_array;
    delete[] ipiv;
}
