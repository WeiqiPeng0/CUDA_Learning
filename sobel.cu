#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

    int idx = y * width + x;

    int gx = -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)]
             + input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];

    int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
             + input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

    int g = abs(gx) + abs(gy);
    g = g > 255 ? 255 : g;
    output[idx] = (unsigned char)g;
}

torch::Tensor sobel_edge(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(input.dim() == 2, "Input must be [H, W]");

    int height = input.size(0);
    int width = input.size(1);

    auto output = torch::zeros_like(input);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    sobel_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<unsigned char>(),
        output.data_ptr<unsigned char>(),
        width, height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}