#include <cuda_runtime.h>
#include <stddef.h>

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Columns of C are along the K dimension
    const size_t col =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Rows of C are along the M dimension
    const size_t row =
        static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row >= static_cast<size_t>(M) ||
        col >= static_cast<size_t>(K)) {
        return;
    }

    float sum = 0.0f;

    // Inner dimension is N
    for (int n = 0; n < N; ++n) {
        sum += A[row * static_cast<size_t>(N) + n] *
               B[static_cast<size_t>(n) * K + col];
    }

    C[row * static_cast<size_t>(K) + col] = sum;
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const dim3 threadsPerBlock(16, 16);
    // Grid covers the output matrix C (M rows × K columns)
    const dim3 blocksPerGrid(
        (K + 15) / 16,
        (M + 15) / 16
    );

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K
    );

    cudaDeviceSynchronize();
}
