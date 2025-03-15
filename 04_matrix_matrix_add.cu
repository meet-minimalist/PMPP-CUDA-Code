#include <iostream>
#include <cuda.h>

__global__
void matrixSum_B(float* A, float* B, float* C, int rows, int cols){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < rows && y < cols){
        int idx = x * cols + y;
        C[idx] = A[idx] + B[idx];
    }
}

__global__
void matrixSum_C(float* A, float* B, float* C, int rows, int cols){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < rows){
        for(int j=0; j<cols; j++){
            int idx = x * cols + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__
void matrixSum_D(float* A, float* B, float* C, int rows, int cols){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(y < cols){
        for(int i=0; i<rows; i++){
            int idx = i * cols + y;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main(){
    int rows = 1000;
    int cols = 5000;

    float* d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, rows * cols * sizeof(float));
    cudaMalloc((void**) &d_B, rows * cols * sizeof(float));
    cudaMalloc((void**) &d_C, rows * cols * sizeof(float));

    dim3 dimGrid(ceil(cols / 32), ceil(rows / 32), 1);
    dim3 dimBlock(32, 32, 1);
    matrixSum_B<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows, cols);
    matrixSum_C<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows, cols);
    matrixSum_D<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaFree((void**) &d_A);
    cudaFree((void**) &d_B);
    cudaFree((void**) &d_C);
    return 0;
}