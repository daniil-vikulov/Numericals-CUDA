#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;

    if (cudaError_t error_id = cudaGetDeviceCount(&deviceCount); error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << error_id << ": " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
    } else {
        std::cout << "Found " << deviceCount << " CUDA-capable device(s)." << std::endl;

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

    return 0;
}
