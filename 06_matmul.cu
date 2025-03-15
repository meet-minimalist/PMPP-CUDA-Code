#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <assert.h>

__global__
void matmul(float *d_a, float *d_b, float *d_c, int N){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= N || y >= N){
        return;
    }
    float res = 0.0f;
    for(int i=0; i<N; i++){
        res += d_a[y * N + i] * d_b[i * N + x];
    }
    d_c[y * N + x] = res;
}

int main(){
    int N = 1024;

    float* h_a = (float*)malloc(N * N * sizeof(float));
    float* h_b = (float*)malloc(N * N * sizeof(float));
    float* h_c = (float*)malloc(N * N * sizeof(float));

    std::fill(h_a, h_a + N * N, 10.0f);
    std::fill(h_b, h_b + N * N, 20.0f);

    float* d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, N * N * (sizeof(float)));
    cudaMalloc((void **) &d_b, N * N * (sizeof(float)));
    cudaMalloc((void **) &d_c, N * N * (sizeof(float)));

    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(float(N) / 16.0f), ceil(float(N) / 16.0f), 1);
    dim3 dimBlock(16, 16, 1);
    matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void **) &d_a);
    cudaFree((void **) &d_b);
    cudaFree((void **) &d_c);

    float res = 0.0f;
    for(int i=0; i<N; i++){
        res += h_a[i] * h_b[i * N];
    }

    std::cout << "Manual ans: " << res << std::endl;
    std::cout << "Computed ans: " << h_c[0] << std::endl;
    assert(res == h_c[0]);

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}