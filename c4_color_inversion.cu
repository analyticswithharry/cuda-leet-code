
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

        // Invert Red, Green, and Blue.
        image[idx]     = 255 - image[idx];
        image[idx + 1] = 255 - image[idx + 1];
        image[idx + 2] = 255 - image[idx + 2];

        // Preserve Alpha: image[idx + 3] remains unchanged.
    }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(
    unsigned char* image,
    int width,
    int height
) {
    int threadsPerBlock = 256;
    int totalPixels = width * height;

    int blocksPerGrid =
        (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        image, width, height
    );

    cudaDeviceSynchronize();
}
