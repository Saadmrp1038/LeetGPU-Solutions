#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < halfN){
        float x1 = input[index];
        float x2 = input[halfN+index];

        float SiLU_x1 = x1 * (1.0f / (1.0f + expf(-x1)));

        float SWiGLU = SiLU_x1 * x2;

        output[index] = SWiGLU;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}