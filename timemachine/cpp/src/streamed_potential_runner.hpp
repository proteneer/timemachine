// this implements a runner for running potentials in multiple streams and then syncing
// with the parent stream
#pragma once

#include "bound_potential.hpp"
#include <memory>
#include <vector>

namespace timemachine {

class StreamedPotentialRunner {

public:
    StreamedPotentialRunner();

    ~StreamedPotentialRunner();

    // wrap execute_device
    void execute_potentials(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        const int N,
        const double *d_x,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_u,
        cudaStream_t stream);

private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;

    // Event for syncing spawned streams with the incoming stream
    cudaEvent_t sync_event_;

    cudaStream_t _get_potential_stream(int index);
    cudaEvent_t _get_potential_event(int index);
};

} // namespace timemachine
