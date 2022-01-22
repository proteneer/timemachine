#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include <cstddef>

namespace timemachine {

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t length) : size(length * sizeof(T)), data_(nullptr) {}

template <typename T> DeviceBuffer<T>::~DeviceBuffer() {
    if (data_) {
        gpuErrchk(cudaFree(data_));
    }
}

template <typename T> void DeviceBuffer<T>::ensure_allocated() {
    if (!data_) {
        gpuErrchk(cudaMalloc(&data_, size));
    } else {
        throw std::runtime_error("attempted to reallocate an allocated buffer");
    }
}

template <typename T> T *DeviceBuffer<T>::data() { return data_; }

template <typename T> void DeviceBuffer<T>::memset(T x) {
    if (data_) {
        gpuErrchk(cudaMemset(data_, x, size));
    } else {
        throw std::runtime_error("called memset on unallocated buffer");
    }
}

template class DeviceBuffer<double>;
template class DeviceBuffer<unsigned long long>;
} // namespace timemachine
