#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "stream_manager.hpp"

namespace timemachine {

StreamManager::StreamManager() {};

StreamManager::~StreamManager() {
    for (const auto &[key, value] : streams_) {
        gpuErrchk(cudaStreamDestroy(value));
    }
    for (const auto &[key, value] : events_) {
        gpuErrchk(cudaEventDestroy(value));
    }
}

cudaStream_t StreamManager::get_stream(int key) {
    if (streams_.count(key) == 1) {
        return streams_[key];
    }
    cudaStream_t new_stream;
    // Create stream that doesn't block with the null stream to avoid unintentional blocking.
    gpuErrchk(cudaStreamCreateWithFlags(&new_stream, cudaStreamNonBlocking));

    streams_[key] = new_stream;
    return new_stream;
};

cudaEvent_t StreamManager::get_stream_event(int key) {
    if (events_.count(key) == 1) {
        return events_[key];
    }
    cudaEvent_t new_event;
    // Create event with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));

    events_[key] = new_event;
    return new_event;
};

// sync_from syncs the managed stream with from_stream
void StreamManager::sync_from(int key, cudaStream_t from_stream) {
    cudaEvent_t event = this->get_stream_event(key);
    gpuErrchk(cudaEventRecord(event, from_stream));
    cudaStream_t to_stream = this->get_stream(key);
    gpuErrchk(cudaStreamWaitEvent(to_stream, event));
};

// sync_to syncs the to_stream from the managed stream
void StreamManager::sync_to(int key, cudaStream_t to_stream) {
    cudaStream_t from_stream = this->get_stream(key);
    cudaEvent_t event = this->get_stream_event(key);
    gpuErrchk(cudaEventRecord(event, from_stream));
    gpuErrchk(cudaStreamWaitEvent(to_stream, event));
};

} // namespace timemachine
