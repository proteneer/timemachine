#pragma once
#include <cstddef>

namespace timemachine {

template <typename T> class DeviceBuffer {
public:
    DeviceBuffer(const size_t length);

    ~DeviceBuffer();

    size_t length;

    size_t size;

    T *data;

    void realloc(const size_t length);

    void copy_from(const T *host_buffer) const;

    void copy_to(T *host_buffer) const;
};

} // namespace timemachine
