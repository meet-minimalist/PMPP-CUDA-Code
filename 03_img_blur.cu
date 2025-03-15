#include <iostream>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_lib/stb_image.h"
#include "stb_lib/stb_image_write.h"


__global__
void imgBlur(unsigned char* img, unsigned char* img_blur, int height, int width, int channels, int blur_kernerl_size){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < width && y < height){
        int blur_delta = (blur_kernerl_size - 1) / 2;
        int num_pixel = 0;
        int pixel_sum = 0;
        for(int c=0; c<channels; c++){
            for(int i=-blur_delta; i<=blur_delta; i++){
                for(int j=-blur_delta; j<=blur_delta; j++){
                    int _img_i = y + i;
                    int _img_j = x + j;
                    if(_img_i < 0 || _img_i >= height || _img_j < 0 || _img_j >= width){
                        continue;
                    }
                    int actual_loc = (_img_i * width + _img_j) * channels + c;
                    pixel_sum += img[actual_loc];
                    num_pixel++;
                }
            }
            int blur_loc = (y * width + x) * channels + c;
            img_blur[blur_loc] = (unsigned char)(pixel_sum / num_pixel);
        }
    }
}

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cerr << "Usage: " << argv[0] << " <img_path> <blur_kernel_size>.\n";
        return -1;
    }

    std::string img_path = argv[1];
    int blur_kernel_size = std::stoi(argv[2]);
    if(blur_kernel_size % 2 == 0){
        std::cerr << "Blur kernel size should be odd number.\n";
        return -1;
    }

    int height, width, channels;
    unsigned char* img = stbi_load(img_path.c_str(), &width, &height, &channels, STBI_rgb);
    if(!img){
        std::cerr << "Failed to load image.\n";
        return -1;
    }
    std::cout << "Image height: " << height << ", width: " << width << ", channels: " << channels << std::endl;

    unsigned char* img_blur = (unsigned char*)malloc(height * width * channels * sizeof(unsigned char));
    unsigned char* d_img, *d_img_blur;
    cudaMalloc((void**) &d_img, height * width * channels * sizeof(unsigned char));
    cudaMalloc((void**) &d_img_blur, height * width * channels * sizeof(unsigned char));

    cudaMemcpy(d_img, img, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(width / 32), ceil(height / 32), 1);
    dim3 dimBlock(32, 32, 1);
    imgBlur<<<dimGrid, dimBlock, 1>>>(d_img, d_img_blur, height, width, channels, blur_kernel_size);
    cudaMemcpy(img_blur, d_img_blur, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree((void **) &d_img);
    cudaFree((void **) &d_img_blur);

    std::string output = "./misc/blur.jpg";
    if(stbi_write_jpg(output.c_str(), width, height, channels, img_blur, 90) == 0){
        std::cerr << "Failed to save output file.\n";
        stbi_image_free(img);
        delete[] img_blur;
        return -1;
    }

    stbi_image_free(img);
    delete[] img_blur;
    return 0;
}