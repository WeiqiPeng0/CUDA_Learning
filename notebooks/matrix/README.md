Pseudocode for tiled matrix multiplication. This saves around S / t times memory accesses, assuming S is total memory access for naiive matmul, and tile size of (t, t).
![](https://github.com/WeiqiPeng0/CUDA_Learning/blob/main/resource/tiled_mm.png)
