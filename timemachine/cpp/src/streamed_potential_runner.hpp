// this implements a runner for running potentials in multiple streams and then syncing
// with the parent stream
#pragma once

#include "bound_potential.hpp"
#include "stream_manager.hpp"
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
        int *d_u_overflow_count,
        cudaStream_t stream);

private:
    StreamManager manager_;
};

} // namespace timemachine
