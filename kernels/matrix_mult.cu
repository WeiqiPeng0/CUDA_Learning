#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// Helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Define tile size - this should be tuned based on your GPU
#define TILE_SIZE 16

__global__
void matrix_mult_tiled_kernel(
    float* output,
    const float* matrix_a,
    const float* matrix_b,
    const int m,  // rows of matrix A
    const int k,  // cols of matrix A / rows of matrix B
    const int n   // cols of matrix B
) {
    // Shared memory for tiles
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    // Calculate global position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        
        // Loop over tiles
        for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
            // Load tile from matrix A
            if (row < m && (tile * TILE_SIZE + threadIdx.x) < k) {
                tile_a[threadIdx.y][threadIdx.x] = matrix_a[row * k + tile * TILE_SIZE + threadIdx.x];
            } else {
                tile_a[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Load tile from matrix B
            if ((tile * TILE_SIZE + threadIdx.y) < k && col < n) {
                tile_b[threadIdx.y][threadIdx.x] = matrix_b[(tile * TILE_SIZE + threadIdx.y) * n + col];
            } else {
                tile_b[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Synchronize threads to ensure tiles are loaded
            __syncthreads();

            // Compute partial sum for this tile
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
            }

            // Synchronize threads before loading next tile
            __syncthreads();
        }
        
        // Write result
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
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);  // TILE_SIZE x TILE_SIZE threads per block
    dim3 number_of_blocks(
        cdiv(n, threads_per_block.x),
        cdiv(m, threads_per_block.y)
    );
    
    // Launch kernel
    matrix_mult_tiled_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<float>(),
        matrix_a.data_ptr<float>(),
        matrix_b.data_ptr<float>(),
        m, k, n
    );
    
    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return result;
} 