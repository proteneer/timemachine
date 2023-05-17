#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "streamed_potential_runner.hpp"

namespace timemachine {

StreamedPotentialRunner::StreamedPotentialRunner() : streams_(0), events_(0) {
    // Setup the event that is used to sync with the incoming stream
    gpuErrchk(cudaEventCreateWithFlags(&sync_event_, cudaEventDisableTiming));
};

StreamedPotentialRunner::~StreamedPotentialRunner() {
    for (int i = 0; i < streams_.size(); i++) {
        gpuErrchk(cudaStreamDestroy(streams_[i]))
    }
    for (int i = 0; i < events_.size(); i++) {
        gpuErrchk(cudaEventDestroy(events_[i]))
    }
    gpuErrchk(cudaEventDestroy(sync_event_));
}

cudaStream_t StreamedPotentialRunner::_get_potential_stream(int index) {
    auto num_streams = streams_.size();
    if (index < num_streams) {
        return streams_[index];
    }
    // Expect stream to be the next increment
    if (num_streams != index) {
        throw std::runtime_error("Asked for new index out of order");
    }
    cudaStream_t new_stream;
    // Create stream that doesn't block with the null stream
    gpuErrchk(cudaStreamCreateWithFlags(&new_stream, cudaStreamNonBlocking));

    cudaEvent_t new_event;
    // Create stream with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));
    streams_.push_back(new_stream);
    events_.push_back(new_event);
    return new_stream;
}

cudaEvent_t StreamedPotentialRunner::_get_potential_event(int index) {
    auto num_events = events_.size();
    if (index >= num_events) {
        throw std::runtime_error("No event with index " + std::to_string(index));
    }
    return events_[index];
}

// wrap execute_device
void StreamedPotentialRunner::execute_potentials(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    const int N,
    const double *d_x,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_u,
    cudaStream_t stream) {
    // Always sync the new streams with the incoming stream to ensure that the state
    // of the incoming buffers are valid
    gpuErrchk(cudaEventRecord(sync_event_, stream));
    for (int i = 0; i < bps.size(); i++) {
        cudaStream_t pot_stream = this->_get_potential_stream(i);

        gpuErrchk(cudaStreamWaitEvent(pot_stream, sync_event_));
        bps[i]->execute_device(N, d_x, d_box, d_du_dx, d_du_dp, d_u, pot_stream);
        cudaEvent_t event = this->_get_potential_event(i);
        // Tell the main stream to synchronize on all of the potential streams
        gpuErrchk(cudaEventRecord(event, pot_stream));
        gpuErrchk(cudaStreamWaitEvent(stream, event));
    }
};

} // namespace timemachine
