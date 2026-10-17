#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619u;
    const unsigned int OFFSET_BASIS = 2166136261u;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(
    const int* input,
    unsigned int* output,
    int N,
    int R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        unsigned int hash = static_cast<unsigned int>(input[idx]);

        for (int round = 0; round < R; round++) {
            hash = fnv1a_hash(hash);
        }

        output[idx] = hash;
    }
}

extern "C" void solve(
    const int* input,
    unsigned int* output,
    int N,
    int R
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, N, R
    );

    cudaDeviceSynchronize();
}
