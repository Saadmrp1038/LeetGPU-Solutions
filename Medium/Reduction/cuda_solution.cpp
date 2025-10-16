#include <cuda_runtime.h>

// Direct atnomicAdd with the double precision would have also worked, but using this technique to reduce the number of atomic operations.
__global__ void reduction_kernel(const float* input, double* output, int N){

    __shared__ double shared_data[256];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (index < N) ? (double)input[index] : 0.0f;
    __syncthreads();

    int range = blockDim.x;

    while(range > 1){
        range >>= 1;

        if(tid < range){
            shared_data[tid] += shared_data[range + tid];
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(output, shared_data[0]);
    }
}

__global__ void cast_double_to_float(double* d_input, float* f_output) {
    *f_output = (float)(*d_input);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  

    double* d_temp_sum;

    cudaMalloc(&d_temp_sum, sizeof(double));
    cudaMemset(d_temp_sum, 0, sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_temp_sum, N);
    cast_double_to_float<<<1, 1>>>(d_temp_sum, output);

    cudaDeviceSynchronize();
    cudaFree(d_temp_sum);
}