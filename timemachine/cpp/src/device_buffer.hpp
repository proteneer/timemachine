#pragma once
#include <cstddef>

namespace timemachine {

template <typename T> class DeviceBuffer {
public:
    DeviceBuffer(const size_t length);

    ~DeviceBuffer();

    const size_t size;

    T *const data;

private:
    T *allocate_(size_t size);
};

} // namespace timemachine
