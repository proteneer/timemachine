#include "observable.hpp"
#include "gpu_utils.cuh"
#include <iostream>


namespace jankmachine {

AvgPartialUPartialParam::AvgPartialUPartialParam(
    BoundPotential *bp, int interval) : bp_(bp), count_(0), interval_(interval) {
    int P = bp_->size();
    gpuErrchk(cudaMalloc(&d_sum_du_dp_, P*sizeof(*d_sum_du_dp_)));
    gpuErrchk(cudaMemset(d_sum_du_dp_, 0, P*sizeof(*d_sum_du_dp_)));
}

AvgPartialUPartialParam::~AvgPartialUPartialParam() {
    gpuErrchk(cudaFree(d_sum_du_dp_));
}

void AvgPartialUPartialParam::observe(
    int step,
    int N,
    double *d_x_t,
    double *d_box_t,
    double lambda) {

    if(step % interval_ == 0) {
        bp_->execute_device(
            N,
            d_x_t,
            d_box_t,
            lambda,
            nullptr,
            d_sum_du_dp_,
            nullptr,
            nullptr,
            static_cast<cudaStream_t>(0) // TBD: parallelize me!
        );
        count_ += 1;
    }

}

void AvgPartialUPartialParam::avg_du_dp(double *h_buf) const {
    gpuErrchk(cudaMemcpy(h_buf, d_sum_du_dp_, this->bp_->size()*sizeof(*h_buf), cudaMemcpyDeviceToHost));
    for(int i=0; i < this->bp_->size(); i++) {
        h_buf[i] /= count_;
    }
}

}