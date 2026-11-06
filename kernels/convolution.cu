#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>

__global__ void convolutionKernel(const unsigned char* d_in, unsigned char* d_out, int width, int height, const float* d_kernel, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height){
        int half = kernelSize / 2;
        float sum = 0.0f;

        for(int ky = -half; ky <= half; ++ky){
            for(int kx = -half; kx <= half; ++kx){
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                float pixelValue = static_cast<float>(d_in[py * width + px]);
                float weight = d_kernel[(ky + half) * kernelSize + (kx + half)];
                sum += pixelValue * weight;
            }
        }
        d_out[y * width + x] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
    }
}

extern "C" void launchCudaConvolution(const unsigned char* h_in, unsigned char* h_out, int width, int height, const std::vector<float>& kernel, int kernelSize){
    size_t imgSize = width * height * sizeof(unsigned char);
    size_t kernelSizeBytes = kernel.size() * sizeof(float);

    unsigned char *d_in = nullptr;
    unsigned char *d_out = nullptr;
    float *d_kernel = nullptr;

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_in, h_in, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    convolutionKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();


    cudaMemcpy(h_out, d_out, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
}