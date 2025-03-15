#include <iostream>
#include <cuda.h>

__global__
void matmulVecMul(float *d_A, float *d_B, float* d_C, int rows, int cols){
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < rows){

        float result = 0.0f;
        for(int j=0; j<cols; j++){
            result += d_A[x * cols + j] * d_B[j];
        }

        d_C[x] = result;
    }
}

int main(){
    int rows = 1000;
    int cols = 5000;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, rows * cols * sizeof(float));
    cudaMalloc((void**) &d_B, cols * sizeof(float));
    cudaMalloc((void**) &d_C, rows * cols * sizeof(float));

    dim3 dimGrid(ceil(cols/32), ceil(rows/32));
    dim3 dimBlock(256);
    matmulVecMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaFree((void**) &d_A);
    cudaFree((void**) &d_B);
    cudaFree((void**) &d_C);
    return 0;
}