
#include <cuda_runtime.h>
#include <cstddef>

__global__ void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    if (N <= 0) {
        return;
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    if (cudaMalloc(&d_A, bytes) != cudaSuccess) return;
    if (cudaMalloc(&d_B, bytes) != cudaSuccess) {
        cudaFree(d_A);
        return;
    }
    if (cudaMalloc(&d_C, bytes) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    if (cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }

    if (err == cudaSuccess) {
        cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
