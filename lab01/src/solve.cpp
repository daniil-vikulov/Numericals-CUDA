#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "gauss.hpp"
#include "iterative_method.hpp"
#include "utils.hpp"
#include "Accelerate/Accelerate.h"


void fillRandomValues(std::vector<double> &arr, double mean = 5.0, double stddev = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);
    for (auto &val: arr) {
        val = dist(gen);
    }
}

double calculateNorm(const std::vector<std::vector<double>> &A, const std::vector<double> &b, const std::vector<double> &x) {
    double norm = 0.0;
    for (size_t i = 0; i < b.size(); i++) {
        double sum = 0;
        for (size_t j = 0; j < A[i].size(); j++) {
            sum += A[i][j] * x[j];
        }
        norm += std::pow(sum - b[i], 2);
    }
    return std::sqrt(norm);
}

void solveWithLAPACK(std::vector<std::vector<double>> &A, std::vector<double> &b, std::vector<double> &x) {
    int n = static_cast<int>(A.size());
    std::vector<double> a_flat;
    for (const auto &row: A) {
        a_flat.insert(a_flat.end(), row.begin(), row.end());
    }
    std::vector<int> ipiv(n);
    int lrm = 101;
    int info = LAPACKE_dgesv(lrm, n, 1, a_flat.data(), n, ipiv.data(), b.data(), 1);
    if (info != 0) {
        std::cerr << "LAPACK solve error, info: " << info << std::endl;
        return;
    }
    x = b;
}

int main() {
    std::cout << "n,lapack_mean,lapack_std,iterative_mean,iterative_std\n";
    const int num_samples = 150;
    std::vector<int> sizes(num_samples);
    for (int i = 0; i < num_samples; i++) {
        sizes[i] = i + 1;
    }

    for (int size: sizes) {
        std::vector<double> results_lapack(100);
        std::vector<double> results_iterative(100);

        for (int i = 0; i < 100; i++) {
            std::vector<std::vector<double>> A(size, std::vector<double>(size));
            std::vector<double> b(size), x(size);
            fillRandomValues(b);

            for (auto &row: A) {
                fillRandomValues(row);
            }

            solveWithLAPACK(A, b, x);
            results_lapack[i] = calculateNorm(A, b, x);

            x.assign(size, 0.0); // Reset x
            iterativeMethod(A, b, x, 20, 1e-5);
            results_iterative[i] = calculateNorm(A, b, x);
        }

        double mean_lapack = std::accumulate(results_lapack.begin(), results_lapack.end(), 0.0) / results_lapack.size();
        double variance_lapack = std::accumulate(results_lapack.begin(), results_lapack.end(), 0.0,
                                                 [mean_lapack](double sum, double value) {
                                                     return sum + (value - mean_lapack) * (value - mean_lapack);
                                                 }) / results_lapack.size();

        double mean_iterative =
                std::accumulate(results_iterative.begin(), results_iterative.end(), 0.0) / results_iterative.size();
        double variance_iterative = std::accumulate(results_iterative.begin(), results_iterative.end(), 0.0,
                                                    [mean_iterative](double sum, double value) {
                                                        return sum +
                                                               (value - mean_iterative) * (value - mean_iterative);
                                                    }) / results_iterative.size();

        std::cout << size << ',' << mean_lapack << ',' << std::sqrt(variance_lapack) << ','
                  << mean_iterative << ',' << std::sqrt(variance_iterative) << '\n';
    }
    return 0;
}