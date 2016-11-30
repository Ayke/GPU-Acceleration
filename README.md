# GPU-Acceleration
A sample code of image processing (more specifically, histogram equalization) with GPU acceleration

## Before compile & run
Please open "CMakeLists.txt", and check:

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_35,code=sm_35)

at the end of this line, I used "sm_35" because I'm using CSIL machine, and its graphics card is GT 720, "3.5" is my Compute Capability, please check your configuration and https://developer.nvidia.com/cuda-gpus to find the suitable compute capability.

## Compile & run
mkdir build && cd build
cmake ..
make
cp /home/XXXX/XXX.ppm in.ppm
cp /home/XXXX/XXX.pgm in.pgm
./5kk70-assignment-gpu
