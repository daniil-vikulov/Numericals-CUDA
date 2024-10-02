#include <format>

#include "CUDASolver.h"

#include <iostream>
#include <random>
#include <algorithm>

void gpuSolve() {
    const std::vector<std::vector<double> > a = {
        {10, 2, 3},
        {4, 20, 5},
        {6, 7, 30}
    };
    const std::vector<double> b = {1, 2, 3};

    int p = 3;

    CUDASolver solver(a, p, b);
    auto x = solver.solve();

    std::cout << "Solution vector x:" << std::endl;
    std::cout << "(";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << std::format("{:.8f}", x[i]);
        if (i < x.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
}

void gpuSolveRandom() {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

    constexpr int n = 10000;
    std::vector<std::vector<double> > a(n, std::vector<double>(n, 0));
    for (auto &row: a) {
        std::ranges::generate(row, [&]() { return static_cast<double>(dist(rng)); });
    }

    std::vector<double> b(n);
    std::ranges::generate(b, [&]() { return static_cast<double>(dist(rng)); });

    CUDASolver solver(a, 1, b);
    auto x = solver.solve();

    std::cout << "Solution vector x:" << std::endl;
    std::cout << "(";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << std::format("{:.8f}", x[i]);
        if (i < x.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
}

int main() {
    gpuSolveRandom();

    return 0;
}
