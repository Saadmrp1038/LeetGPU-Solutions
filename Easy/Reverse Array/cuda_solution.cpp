#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N/2){
        float tmp = input[index];
        input[index] = input[N-1-index];
        input[N-1-index] = tmp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}