# GPU-Acceleration
A sample code of image processing (more specifically, histogram equalization) with GPU acceleration

## Before compile & run
Please open "CMakeLists.txt", and check:

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_52)

at the end of this line, I used "sm_52" because I'm using GTX 960, that's my Compute Capability, please check your configuration and https://developer.nvidia.com/cuda-gpus to find the suitable compute capability.

