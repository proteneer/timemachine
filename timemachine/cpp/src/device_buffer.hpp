#pragma once
#include <cstdlib>

namespace timemachine {

template <typename T> class DeviceBuffer {

public:
    DeviceBuffer(const std::size_t length);

    ~DeviceBuffer();

    const size_t size;

    void ensure_allocated();

    T *data();

    void memset(T x);

private:
    T *data_;
};

} // namespace timemachine
