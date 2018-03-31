# GPU-Acceleration
This project shows a sample code of processing images with GPU acceleration, more specifically, *histogram equlization*.

## Before compile
Set correct compile configruation

Edit `CMakeLists.txt`, and check this line `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_35,code=sm_35)`
, the configuration `sm_35` matches graphics card **GT 720**. Please use this [link](https://developer.nvidia.com/cuda-gpus) to check and find a suitable compute capability configurations.

## Compile & run
To compile, use the following script

```sh
mkdir build && cd build
cmake ..
make
```


Example of running this project

```sh
cp ~/the/path/of/a/image.ppm in.ppm
./5kk70-assignment-gpu
```
This project only accepts `ppm` or `pgm` file, and input name must be `in.ppm` or `in.pgm`.
