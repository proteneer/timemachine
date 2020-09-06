#include <vector>

#include "observable.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

AvgPartialUPartialTheta::AvgPartialUPartialTheta(
    std::vector<BoundPotential *> bps
) : bps_(bps) {

    // create buffer for du/dp derivatives
    for(int i=0; i < bps_.size(); i++) {
        int P = bps_[i]->size();
        double *du_dp;
        gpuErrchk(cudaMalloc(&du_dp, P*sizeof(*du_dp)));
        gpuErrchk(cudaMemset(du_dp, 0, P*sizeof(*du_dp)));
        d_du_dp_.push_back(du_dp);
    }

}

void AvgPartialUPartialTheta::collect(
    int step,
    int N,
    double *d_x_t,
    double *d_box_t,
    double lambda) {

    for(int i=0; i < bps_.size(); i++) {

        bps_[i]->execute_device(
            N,
            d_x_t,
            d_box_t,
            lambda,
            nullptr,
            d_du_dp_[i],
            nullptr,
            nullptr,
            static_cast<cudaStream_t>(0) // TBD: parallelize me!
        );

    }

}

}