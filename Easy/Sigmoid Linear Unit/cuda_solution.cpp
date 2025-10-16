#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N){
        float x = input[index];
        output[index] = x * (1.0f / (1.0f + expf(-x)));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

