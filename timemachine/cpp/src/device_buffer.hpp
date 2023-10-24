#pragma once
#include <cstddef>

namespace timemachine {

template <typename T> class DeviceBuffer {
public:
    DeviceBuffer();
    DeviceBuffer(const size_t length);

    ~DeviceBuffer();

    size_t length;

    T *data;

    void realloc(const size_t length);

    size_t size() const;

    void copy_from(const T *host_buffer) const;

    void copy_to(T *host_buffer) const;
};

} // namespace timemachine
