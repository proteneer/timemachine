#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include <cstddef>

namespace timemachine {

template <typename T> T *allocate(const std::size_t length) {
    T *buffer;
    gpuErrchk(cudaMalloc(&buffer, length * sizeof(T)));
    return buffer;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t length) : size(length * sizeof(T)), data(allocate<T>(length)) {}

template <typename T> DeviceBuffer<T>::~DeviceBuffer() { gpuErrchk(cudaFree(data)); }

template class DeviceBuffer<double>;
template class DeviceBuffer<unsigned long long>;
} // namespace timemachine
