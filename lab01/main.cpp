#include <format>

#include "CUDASolver.h"

#include <iostream>

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

int main() {
    gpuSolve();

    return 0;
}
