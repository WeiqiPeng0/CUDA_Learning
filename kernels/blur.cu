#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__device__ const float gaussian_kernel[3][3] = {
    {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
    {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
    {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
};

__global__
void blur_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        for (int c = 0; c < channels; c++) {
            float blur_value = 0.0f;
            
            // Apply 3x3 gaussian kernel
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int cur_row = min(max(row + i, 0), height - 1);
                    int cur_col = min(max(col + j, 0), width - 1);
                    int input_idx = (cur_row * width + cur_col) * channels + c;
                    
                    blur_value += (float)input[input_idx] * gaussian_kernel[i+1][j+1];
                }
            }
            
            int output_idx = (row * width + col) * channels + c;
            output[output_idx] = (unsigned char)blur_value;
        }
    }
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor blur_image(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                         cdiv(height, threads_per_block.y));

    blur_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );

    // check CUDA error status
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
} 