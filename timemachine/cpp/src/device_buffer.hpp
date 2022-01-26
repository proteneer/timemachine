#pragma once
#include <cstddef>

namespace timemachine {

template <typename T> class DeviceBuffer {
public:
    DeviceBuffer(const size_t length);

    ~DeviceBuffer();

    const size_t size;

    T *const data;

    void copy_from(const T *host_buffer) const;

    void copy_to(T *host_buffer) const;
};

} // namespace timemachine
