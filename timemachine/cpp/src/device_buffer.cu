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

template <typename T> DeviceBuffer<T>::~DeviceBuffer() {
    // TODO: the file/line context reported by gpuErrchk on failure is
    // not very useful when it's called from here. Is there a way to
    // report a stack trace?
    gpuErrchk(cudaFree(data));
}

template <typename T> void DeviceBuffer<T>::copy_from(const T *host_buffer) const {
    gpuErrchk(cudaMemcpy(data, host_buffer, size, cudaMemcpyHostToDevice));
}

template <typename T> void DeviceBuffer<T>::copy_to(T *host_buffer) const {
    gpuErrchk(cudaMemcpy(host_buffer, data, size, cudaMemcpyDeviceToHost));
}

template class DeviceBuffer<double>;
template class DeviceBuffer<unsigned long long>;
} // namespace timemachine
