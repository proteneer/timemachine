#include "observable.hpp"
#include "gpu_utils.cuh"
#include <iostream>


namespace timemachine {

AvgPartialUPartialParam::AvgPartialUPartialParam(
    BoundPotential *bp, int freq) : bp_(bp), count_(0), freq_(freq) {
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

    if(step % freq_ == 0) {
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

void AvgPartialUPartialParam::avg_du_dp(double *h_buf) {
    gpuErrchk(cudaMemcpy(h_buf, d_sum_du_dp_, this->bp_->size()*sizeof(*h_buf), cudaMemcpyDeviceToHost));
    for(int i=0; i < this->bp_->size(); i++) {
        h_buf[i] /= count_;
    }
}


AvgPartialUPartialLambda::AvgPartialUPartialLambda(
    std::vector<BoundPotential *>bps, int freq) : bps_(bps), count_(0), freq_(freq) {
    gpuErrchk(cudaMalloc(&d_sum_du_dl_, 1*sizeof(*d_sum_du_dl_)));
    gpuErrchk(cudaMemset(d_sum_du_dl_, 0, 1*sizeof(*d_sum_du_dl_)));
}

AvgPartialUPartialLambda::~AvgPartialUPartialLambda() {
    gpuErrchk(cudaFree(d_sum_du_dl_));
}

void AvgPartialUPartialLambda::observe(
    int step,
    int N,
    double *d_x_t,
    double *d_box_t,
    double lambda) {

    if(step % freq_ == 0) {
        for(int i=0; i < bps_.size(); i++) {
            bps_[i]->execute_device(
                N,
                d_x_t,
                d_box_t,
                lambda,
                nullptr,
                nullptr,
                d_sum_du_dl_,
                nullptr,
                static_cast<cudaStream_t>(0) // TBD: parallelize me!
            );
        }
        count_ += 1;
    }

}

void AvgPartialUPartialLambda::avg_du_dl(double *h_buf) {
    gpuErrchk(cudaMemcpy(h_buf, d_sum_du_dl_, 1*sizeof(*h_buf), cudaMemcpyDeviceToHost));
    h_buf[0] /= count_;
}


}