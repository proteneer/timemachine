#include "device_buffer.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

void *allocate(const std::size_t size) {
    void *buffer;
    gpuErrchk(cudaMalloc(&buffer, size));
    return buffer;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t length)
    : size(length * sizeof(T)), data(static_cast<T *>(allocate(size))) {}

template <typename T> DeviceBuffer<T>::~DeviceBuffer() { gpuErrchk(cudaFree(data)); }

template <typename T> void DeviceBuffer<T>::copy_from_host(const T *host_buffer) const {
    gpuErrchk(cudaMemcpy(data, host_buffer, size, cudaMemcpyHostToDevice));
}

template <typename T> void DeviceBuffer<T>::copy_to_host(T *host_buffer) const {
    gpuErrchk(cudaMemcpy(host_buffer, data, size, cudaMemcpyDeviceToHost));
}

template <typename T> void DeviceBuffer<T>::copy_from_device(const T *device_buffer) const {
    gpuErrchk(cudaMemcpy(data, device_buffer, size, cudaMemcpyDeviceToDevice));
}

template <typename T> void DeviceBuffer<T>::copy_to_device(T *device_buffer) const {
    gpuErrchk(cudaMemcpy(device_buffer, data, size, cudaMemcpyDeviceToDevice));
}

template <typename T> void DeviceBuffer<T>::copy_from_host_async(const T *host_buffer, cudaStream_t stream) const {
    gpuErrchk(cudaMemcpyAsync(data, host_buffer, size, cudaMemcpyHostToDevice, stream));
}

template <typename T> void DeviceBuffer<T>::copy_to_host_async(T *host_buffer, cudaStream_t stream) const {
    gpuErrchk(cudaMemcpyAsync(host_buffer, data, size, cudaMemcpyDeviceToHost, stream));
}

template <typename T> void DeviceBuffer<T>::copy_from_device_async(const T *device_buffer, cudaStream_t stream) const {
    gpuErrchk(cudaMemcpyAsync(data, device_buffer, size, cudaMemcpyDeviceToDevice, stream));
}

template <typename T> void DeviceBuffer<T>::copy_to_device_async(T *device_buffer, cudaStream_t stream) const {
    gpuErrchk(cudaMemcpyAsync(device_buffer, data, size, cudaMemcpyDeviceToDevice, stream));
}

template class DeviceBuffer<char>;
template class DeviceBuffer<int>;
template class DeviceBuffer<unsigned int>;
template class DeviceBuffer<unsigned long long>;
template class DeviceBuffer<double>;
} // namespace timemachine
