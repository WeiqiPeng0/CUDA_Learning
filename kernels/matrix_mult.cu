#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// Helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__
void matrix_mult_kernel(
    float* output,
    const float* matrix_a,
    const float* matrix_b,
    const int m,  // rows of matrix A
    const int k,  // cols of matrix A / rows of matrix B
    const int n   // cols of matrix B
) {
    // Calculate global position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        
        // Compute dot product for this element
        for (int i = 0; i < k; i++) {
            sum += matrix_a[row * k + i] * matrix_b[i * n + col];
        }
        
        output[row * n + col] = sum;
    }
}

torch::Tensor matrix_mult(torch::Tensor matrix_a, torch::Tensor matrix_b) {
    // Check if tensors are on CUDA
    assert(matrix_a.device().type() == torch::kCUDA);
    assert(matrix_b.device().type() == torch::kCUDA);
    
    // Check if tensors are float32
    assert(matrix_a.dtype() == torch::kFloat32);
    assert(matrix_b.dtype() == torch::kFloat32);
    
    // Get dimensions
    const auto m = matrix_a.size(0);  // rows of matrix A
    const auto k = matrix_a.size(1);  // cols of matrix A / rows of matrix B
    const auto n = matrix_b.size(1);  // cols of matrix B
    
    // Verify dimensions are compatible for multiplication
    assert(matrix_b.size(0) == k);
    
    // Create output tensor
    auto result = torch::empty({m, n}, matrix_a.options());
    
    // Define block and grid dimensions
    dim3 threads_per_block(16, 16);  // 256 threads per block
    dim3 number_of_blocks(
        cdiv(n, threads_per_block.x),
        cdiv(m, threads_per_block.y)
    );
    
    // Launch kernel
    matrix_mult_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<float>(),
        matrix_a.data_ptr<float>(),
        matrix_b.data_ptr<float>(),
        m, k, n
    );
    
    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return result;
} 