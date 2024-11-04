#include "iterative_method.hpp"
#include "tests.hpp"
#include <iostream>

void testIterativeMethod() {
    std::vector<std::vector<double>> A = {{4, 1, 1}, {1, 3, 0}, {1, 0, 2}};
    std::vector<double> b = {6, 5, 4};
    std::vector<double> x(3);

    int max_iters = 100;
    double delta = 1e-10;

    int result = iterativeMethod(A, b, x, max_iters, delta);

    if (result == 0) {
        std::cout << "Solution found:\n";
        for (const auto& value : x) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "Solution did not converge.\n";
    }
}

int testIterativeMethodSparse() {
    std::vector<double> values = {4, 3, 1, 3, 2};
    std::vector<int> row_indices = {0, 0, 1, 1, 2};
    std::vector<int> col_ptrs = {0, 2, 4, 5};
    std::vector<double> b = {6, 5, 9};
    std::vector<double> x(3);

    int max_iters = 100;
    double tolerance = 1e-10;

    int result =
        iterativeMethodSparse(values, row_indices, col_ptrs, b, x, max_iters, tolerance, 3);

    if (result == 0) {
        std::cout << "Solution found:\n";
        for (const auto& value : x) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "Solution did not converge.\n";
    }

    return 0;
}

void runTests() {
    testIterativeMethod();
    // testIterativeMethodSparse();
}
