#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include <cstddef>

namespace timemachine {

template <typename T, bool PINNED> T *allocate(const std::size_t length) {
    if (length < 1) {
        throw std::runtime_error("device buffer length must at least be 1");
    }
    T *buffer;
    if (PINNED) {
        gpuErrchk(cudaMallocHost(&buffer, length * sizeof(T)));
    } else {
        gpuErrchk(cudaMalloc(&buffer, length * sizeof(T)));
    }
    return buffer;
}

template <typename T, bool PINNED>
DeviceBuffer<T, PINNED>::DeviceBuffer(const std::size_t length)
    : size(length * sizeof(T)), data(allocate<T, PINNED>(length)) {}

template <typename T, bool PINNED> DeviceBuffer<T, PINNED>::~DeviceBuffer() {
    // TODO: the file/line context reported by gpuErrchk on failure is
    // not very useful when it's called from here. Is there a way to
    // report a stack trace?
    if (PINNED) {
        gpuErrchk(cudaFreeHost(data));
    } else {
        gpuErrchk(cudaFree(data));
    }
}

template <typename T, bool PINNED> void DeviceBuffer<T, PINNED>::copy_from(const T *host_buffer) const {
    if (PINNED) {
        memcpy(data, host_buffer, size);
    } else {
        gpuErrchk(cudaMemcpy(data, host_buffer, size, cudaMemcpyHostToDevice));
    }
}

template <typename T, bool PINNED> void DeviceBuffer<T, PINNED>::copy_to(T *host_buffer) const {
    if (PINNED) {
        memcpy(host_buffer, data, size);
    } else {
        gpuErrchk(cudaMemcpy(host_buffer, data, size, cudaMemcpyDeviceToHost));
    }
}

template class DeviceBuffer<double, true>;
template class DeviceBuffer<double, false>;
template class DeviceBuffer<int, true>;
template class DeviceBuffer<int, false>;
template class DeviceBuffer<char, true>;
template class DeviceBuffer<char, false>;
template class DeviceBuffer<unsigned int, true>;
template class DeviceBuffer<unsigned int, false>;
template class DeviceBuffer<unsigned long long, true>;
template class DeviceBuffer<unsigned long long, false>;
} // namespace timemachine
