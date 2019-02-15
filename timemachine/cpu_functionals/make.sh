PLATFORM_FLAGS="";
if [ "$(uname)" == "Darwin" ]; then
	PLATFORM_FLAGS="-undefined dynamic_lookup";
    # Do something under Mac OS X platform        
fi

# g++ -g -O0 -march=native -Wall -shared -std=c++11 -fPIC $PLATFORM_FLAGS `python3 -m pybind11 --includes` wrappers.cpp -o energy`python3-config --extension-suffix`

# g++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $PLATFORM_FLAGS `python3 -m pybind11 --includes` wrappers.cpp -o energy`python3-config --extension-suffix`
# nvcc -Xcompiler="-fPIC" -shared -std=c++11 $PLATFORM_FLAGS `python3 -m pybind11 --includes` wrappers.cpp -o energy`python3-config --extension-suffix`
# nvcc  -Xcompiler="-fPIC -std=c++11" -O3 -shared -std=c++11 $PLATFORM_FLAGS `python3 -m pybind11 --includes` integrator_wrappers.cu -o energy`python3-config --extension-suffix`


nvcc --use_fast_math --ptxas-options=-v -lineinfo -arch=sm_72 -O3 -Xcompiler="-fPIC" -std=c++11 integrator.cu nonbonded_gpu.cu bonded_gpu.cu gpu_utils.cu -c

g++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $PLATFORM_FLAGS `python3 -m pybind11 --includes` -L/usr/local/cuda-10.0/lib64/ -I/usr/local/cuda-10.0/include/ wrappers.cpp integrator.o nonbonded_gpu.o bonded_gpu.o gpu_utils.o -o custom_ops`python3-config --extension-suffix` -lcurand -lcublas -lcudart
