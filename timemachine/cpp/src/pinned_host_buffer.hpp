#pragma once
#include <cstddef>

namespace timemachine {

template <typename T> class PinnedHostBuffer {
public:
    PinnedHostBuffer(const size_t length);

    ~PinnedHostBuffer();

    const size_t size;

    T *const data;

    void copy_from(const T *host_buffer) const;

    void copy_to(T *host_buffer) const;
};

} // namespace timemachine
