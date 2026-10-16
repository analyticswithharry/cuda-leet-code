#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = N - 1 - i;

    if (i < N / 2) {
        float temp = input[i];
        input[i] = input[j];
        input[j] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
}
