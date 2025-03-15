#include <iostream>
#include <cuda.h>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_lib/stb_image.h"
#include "stb_lib/stb_image_write.h"

// Compilation command: nvcc -o main 02_rgb_gray.cu -I./stb_lib/

__global__
void rgbToGray(unsigned char* img, unsigned char* gray_img, int height, int width, int channels){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < height && col < width){
        int gray_idx = row * width + col;
        int rgb_idx = gray_idx * channels;

        unsigned char r = img[rgb_idx];
        unsigned char g = img[rgb_idx + 1];
        unsigned char b = img[rgb_idx + 2];
        gray_img[gray_idx] = (unsigned char)(r * 0.21f + g * 0.72f + b * 0.07f);
    }
}

int main(int argc, char* argv[]){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return 1;
    }

    std::string img_path = argv[1];

    // Query the properties of the first device (you can iterate if you have multiple devices)
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Query device 0

    // Print the maximum number of threads per block
    std::cout << "Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum block dimensions (x, y, z): "
              << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << std::endl;

    std::cout << "Maximum grid dimensions (x, y, z): "
              << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << std::endl;

    int height, width, channels;
    unsigned char* img = stbi_load(img_path.c_str(), &width, &height, &channels, STBI_rgb);
    if(img == nullptr){
        std::cerr << "Image loading failure.\n";
        return 1;
    }
    std::cout << "Image height: " << height << ", width: " << width << ", channels: " << channels << std::endl;

    unsigned char* gray_img = (unsigned char*)malloc(height * width * sizeof(unsigned char));

    dim3 dimGrid(ceil(width / 32), ceil(height / 32), 1);
    dim3 dimBlock(32, 32, 1);

    unsigned char* d_img;
    unsigned char* d_gray_img;
    cudaMalloc((void**) &d_img, height * width * channels * sizeof(unsigned char));
    cudaMalloc((void**) &d_gray_img, height * width * sizeof(unsigned char));

    cudaMemcpy(d_img, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    rgbToGray<<<dimGrid, dimBlock>>>(d_img, d_gray_img, height, width, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(gray_img, d_gray_img, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::string output = "./misc/gray.jpg";
    if(stbi_write_jpg(output.c_str(), width, height, 1, gray_img, width * height) == 0){
        std::cerr << "Failed to save the image." << std::endl;
        stbi_image_free(img);
        delete[] gray_img;
        return -1;
    }
    cudaFree((void**) &d_img);
    cudaFree((void**) &d_gray_img);
    stbi_image_free(img);
    delete[] gray_img;
    return 0;
}