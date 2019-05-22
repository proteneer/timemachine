nvcc -arch=sm_61 -G -g -Xcompiler -fPIC -I ~/Code/timemachine/timemachine/cpu_functionals/ gpu/gpu_bonded_kernels.cu -c

g++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $PLATFORM_FLAGS `python3 -m pybind11 --includes` -L/usr/local/cuda-10.1/lib64/ -I/usr/local/cuda-10.1/include/ wrap_kernels.cpp gpu_bonded_kernels.o -o custom_ops`python3-config --extension-suffix` -lcurand -lcublas -lcudart
