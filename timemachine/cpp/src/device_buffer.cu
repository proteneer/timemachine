#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include <cstddef>

namespace timemachine {

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t length) : size(length * sizeof(T)), data(allocate_(size)) {}

template <typename T> DeviceBuffer<T>::~DeviceBuffer() { gpuErrchk(cudaFree(data)); }

template <typename T> void DeviceBuffer<T>::memset(T x) { gpuErrchk(cudaMemset(data, x, size)); }

template <typename T> T *DeviceBuffer<T>::allocate_(const std::size_t size) {
    T *data;
    gpuErrchk(cudaMalloc(&data, size));
    return data;
}

template class DeviceBuffer<double>;
template class DeviceBuffer<unsigned long long>;
} // namespace timemachine
