#include <iostream>
#include <cuda.h>
#include <algorithm>

#define TILE_WIDTH 2

__global__
void tiledMatmul(float* d_a, float* d_b, float* d_c, int N){
    int x = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int y = TILE_WIDTH * blockIdx.y + threadIdx.y;

    // blockDim.x and blockDim.y are not compile time constants. Hence we can't use them to define a_s.
    // Hence need to explicitly define TILE_WIDTH and refer that inplace of blockDim.x and blockDim.y.
    __shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

    int num_phases = ceil(float(N) / float(TILE_WIDTH));
    // since we are dealing with N x N matrices only, num_phases will be same for a and b.

    float res = 0.0f;
    for(int i=0; i<num_phases; i++){
        a_s[threadIdx.y][threadIdx.x] = d_a[y * N + (i * TILE_WIDTH + threadIdx.x)];
        b_s[threadIdx.y][threadIdx.x] = d_b[(i * TILE_WIDTH + threadIdx.y) * N + x];

        __syncthreads();
        for(int k=0; k<TILE_WIDTH; k++){
            res += a_s[threadIdx.y][k] * b_s[k][threadIdx.x];
        }
        __syncthreads();
    }
    d_c[y * N + x] += res;
}

void cpuMatmul(float* h_a, float* h_b, float* h_c, int N){
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
            float res = 0.0f;
            for(int k=0; k<N; k++){
                res += h_a[j*N + k] * h_b[k*N + i];
            }
            h_c[j*N + i] = res;
        }
    }
}


bool compare(float* h_c_gpu, float* h_c_cpu, int N){
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
            if(h_c_gpu[j*N + i] != h_c_cpu[j*N + i]){
                std::cout << "Mismatch at x: " << i << ", y: " << j << ", CPU: " << h_c_cpu[j*N + i] << ", GPU: " << h_c_gpu[j*N + i] << std::endl;
            }
        }
    }
    return true;
}

int main(){
    int N = 1024;

    float* h_a = (float*)malloc(N * N * sizeof(float));
    float* h_b = (float*)malloc(N * N * sizeof(float));
    float* h_c = (float*)malloc(N * N * sizeof(float));
    float* h_c_cpu = (float*)malloc(N * N * sizeof(float));

    for(int i=0; i<N*N; i++){
        h_c_cpu[i] = h_c[i];
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, N * N * sizeof(float));
    cudaMalloc((void **) &d_b, N * N * sizeof(float));
    cudaMalloc((void **) &d_c, N * N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(float(N)/float(TILE_WIDTH)), ceil(float(N)/float(TILE_WIDTH)), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    std::cout << "Tiled Matmul started." << std::endl;
    tiledMatmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
    std::cout << "Tiled Matmul completed." << std::endl;
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void **) &d_a);
    cudaFree((void **) &d_b);
    cudaFree((void **) &d_c);

    std::cout << "CPU started." << std::endl;
    cpuMatmul(h_a, h_b, h_c_cpu, N);
    std::cout << "CPU completed." << std::endl;

    bool status = compare(h_c, h_c_cpu, N);
    std::cout << "Results matching." << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    return 0;
}