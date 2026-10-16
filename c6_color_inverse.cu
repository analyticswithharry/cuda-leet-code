
#include <cuda_runtime.h>

__global__ void invert_kernel(
    unsigned char* image,
    int width,
    int height
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (pixel < totalPixels) {
        int idx = pixel * 4;

        image[idx]     = 255 - image[idx];      // Red
        image[idx + 1] = 255 - image[idx + 1];  // Green
        image[idx + 2] = 255 - image[idx + 2];  // Blue

        // Alpha remains unchanged.
    }
}

extern "C" void solve(
    unsigned char* image,
    int width,
    int height
) {
    int totalPixels = width * height;
    int threadsPerBlock = 256;

    int blocksPerGrid =
        (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        image, width, height
    );
}
