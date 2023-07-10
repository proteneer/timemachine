#include "streamed_potential_runner.hpp"

namespace timemachine {

StreamedPotentialRunner::StreamedPotentialRunner(){};

StreamedPotentialRunner::~StreamedPotentialRunner() {}

// wrap execute_device
void StreamedPotentialRunner::execute_potentials(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    const int N,
    const double *d_x,   // [N * 3]
    const double *d_box, // [3 * 3]
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_u, // [bps.size()]
    int *d_u_overflow_count, // [1]
    cudaStream_t stream) {
    for (int i = 0; i < bps.size(); i++) {
        // Always sync the new streams with the incoming stream to ensure that the state
        // of the incoming buffers are valid
        manager_.sync_from(i, stream);
    }
    for (int i = 0; i < bps.size(); i++) {
        bps[i]->execute_device(
            N,
            d_x,
            d_box,
            d_du_dx,
            d_du_dp,
            d_u == nullptr ? nullptr : d_u + i,
            d_u_overflow_count,
            manager_.get_stream(i));
        manager_.sync_to(i, stream);
    }
};

} // namespace timemachine
