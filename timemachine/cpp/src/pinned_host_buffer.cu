#include "gpu_utils.cuh"
#include "pinned_host_buffer.hpp"

namespace timemachine {

template <typename T> T *allocate(const std::size_t length) {
    if (length < 1) {
        throw std::runtime_error("device buffer length must at least be 1");
    }
    T *buffer;
    gpuErrchk(cudaMallocHost(&buffer, length * sizeof(T)));
    return buffer;
}

template <typename T>
PinnedHostBuffer<T>::PinnedHostBuffer(const std::size_t length) : size(length * sizeof(T)), data(allocate<T>(length)) {}

template <typename T> PinnedHostBuffer<T>::~PinnedHostBuffer() {
    // TODO: the file/line context reported by gpuErrchk on failure is
    // not very useful when it's called from here. Is there a way to
    // report a stack trace?
    gpuErrchk(cudaFreeHost(data));
}

template <typename T> void PinnedHostBuffer<T>::copy_from(const T *host_buffer) const {
    memcpy(data, host_buffer, size);
}

template <typename T> void PinnedHostBuffer<T>::copy_to(T *host_buffer) const { memcpy(host_buffer, data, size); }

template class PinnedHostBuffer<double>;
template class PinnedHostBuffer<float>;
template class PinnedHostBuffer<int>;
template class PinnedHostBuffer<char>;
template class PinnedHostBuffer<unsigned int>;
template class PinnedHostBuffer<unsigned long long>;
} // namespace timemachine
