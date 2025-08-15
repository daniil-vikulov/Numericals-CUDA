#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <cblas.h>
#include <vector>

void checkCuda() {
    int deviceCount = 0;

    assert(cudaGetDeviceCount(&deviceCount) == cudaSuccess);

    assert(deviceCount > 0);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    }
}

void checkBlas() {
    int n = 3;
    std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<double> C(n * n, 0);

    // Perform C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);

    std::cout << "Result matrix C:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    checkCuda();
    checkBlas();

    return 0;
}
