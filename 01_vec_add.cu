#include <iostream>
#include <cuda.h>

void vecAdd(float* a, float* b, float* c, int num_elems){
    for(int i=0; i<num_elems; i++){
        c[i] = a[i] + b[i];
    }
}

__global__
void vecAddKernel(float* d_a, float* d_b, float* d_c, int num_elems){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < num_elems){
        d_c[i] = d_a[i] + d_b[i];
    }
}

void vecAddCuda(float* h_a, float* h_b, float* h_c, int num_elems){
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, num_elems * sizeof(float));
    cudaMalloc((void**) &d_b, num_elems * sizeof(float));
    cudaMalloc((void**) &d_c, num_elems * sizeof(float));

    cudaMemcpy(d_a, h_a, num_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_elems * sizeof(float), cudaMemcpyHostToDevice);
    vecAddKernel<<<ceil(num_elems / 256.0), 256>>>(d_a, d_b, d_c, num_elems);

    cudaMemcpy(h_c, d_c, num_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(){
    int total_num = 1000000;
    float* h_a = (float*)malloc(total_num * sizeof(float));
    float* h_b = (float*)malloc(total_num * sizeof(float));
    float* h_c = (float*)malloc(total_num * sizeof(float));

    for(int i=0; i<total_num; i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    vecAddCuda(h_a, h_b, h_c, total_num);
    std::cout << "Result's last element: " << h_c[total_num-1] << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
