Pseudocode for tiled matrix multiplication. This saves around S / t times memory accesses, assuming S is total memory access for naiive matmul, and tile size of (t, t).
![](https://github.com/WeiqiPeng0/CUDA_Learning/blob/main/resource/tiled_mm.png)


Reference:
- https://llmsystem.github.io/llmsystem2024spring/assets/files/11868_LLM_Systems_Assignment_1-ee1244bd2b2f8f2de8e5e9e857f6791f.pdf
