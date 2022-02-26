#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>

namespace timemachine {

template <typename T> class DeviceBuffer {
public:
    DeviceBuffer(const std::size_t length);

    // Disable the copy constructor.
    // A correct implementation of this would allocate a new buffer
    // and copy data, but we generally don't want to do this.
    // Disabling copies prevents accidental pass-by-value
    DeviceBuffer(const DeviceBuffer &) = delete;

    ~DeviceBuffer();

    const std::size_t size;

    T *const data;

    void copy_from_host(const T *host_buffer) const;

    void copy_to_host(T *host_buffer) const;

    void copy_from_device(const T *device_buffer) const;

    void copy_to_device(T *device_buffer) const;

    // async API

    void copy_from_host_async(const T *host_buffer, cudaStream_t stream) const;

    void copy_to_host_async(T *host_buffer, cudaStream_t stream) const;

    void copy_from_device_async(const T *device_buffer, cudaStream_t stream) const;

    void copy_to_device_async(T *device_buffer, cudaStream_t stream) const;
};

} // namespace timemachine
