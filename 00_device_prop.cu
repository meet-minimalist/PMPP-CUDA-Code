#include <cuda_runtime.h>
#include <iostream>

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "  Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    std::cout << "  Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "  Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "  L2 Cache Size: " << prop.l2CacheSize / 1024.0 << " KB" << std::endl;
    std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  ECC Support: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Registers per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        printDeviceProperties(i);
    }

    return 0;
}
