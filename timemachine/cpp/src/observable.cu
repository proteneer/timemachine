#include "observable.hpp"
#include "gpu_utils.cuh"
#include <iostream>


namespace timemachine {

void __global__ k_compute_variance(
    const int N,
    const int k,
    const double * __restrict__ d_du_dp,
    double *d_m,
    double *d_s
    ) {

    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= N) {
        return;
    }

    if(d_du_dp) {
        // Taken from https://www.johndcook.com/blog/standard_deviation/
        double x = d_du_dp[idx];
        if (k == 1) {
            d_m[idx] = x;
            d_s[idx] = 0;
        } else {
            const double m_old = d_m[idx];
            const double m_new = m_old + (x - m_old) / k;
            d_m[idx] = m_new;
            d_s[idx] += (x - m_old) * (x - m_new);
        }
    }
}

AvgPartialUPartialParam::AvgPartialUPartialParam(
    BoundPotential *bp, int interval) : bp_(bp), count_(0), interval_(interval) {
    int P = bp_->size();
    gpuErrchk(cudaMalloc(&d_du_dp_, P*sizeof(*d_du_dp_)));
    gpuErrchk(cudaMalloc(&d_m_, P*sizeof(*d_m_)));
    gpuErrchk(cudaMalloc(&d_s_, P*sizeof(*d_s_)));
}

AvgPartialUPartialParam::~AvgPartialUPartialParam() {
    gpuErrchk(cudaFree(d_du_dp_));
    gpuErrchk(cudaFree(d_m_));
    gpuErrchk(cudaFree(d_s_));
}

void AvgPartialUPartialParam::observe(
    int step,
    int N,
    double *d_x_t,
    double *d_box_t,
    double lambda) {

    if(step % interval_ == 0) {
        cudaStream_t stream = static_cast<cudaStream_t>(0);
        const int size = bp_->size();
        // Need the latest du_dp, so reset to zero each round
        gpuErrchk(cudaMemsetAsync(d_du_dp_, 0, size*sizeof(*d_du_dp_), stream));
        bp_->execute_device(
            N,
            d_x_t,
            d_box_t,
            lambda,
            nullptr,
            d_du_dp_,
            nullptr,
            nullptr,
            stream // TBD: parallelize me!
        );
        count_++;
        const int tpb = 32;
        const int blocks = (size+tpb-1)/tpb;
        k_compute_variance<<<blocks, tpb, 0, stream>>>(
            size,
            count_,
            d_du_dp_,
            d_m_,
            d_s_
        );

        gpuErrchk(cudaPeekAtLastError());
    }

}

void AvgPartialUPartialParam::std_du_dp(double *h_buf) const {
    // Copying is only necessary if there is a variance to return
    if (count_ > 1) {
        gpuErrchk(cudaMemcpy(h_buf, d_s_, this->bp_->size()*sizeof(*h_buf), cudaMemcpyDeviceToHost));
    }
    for(int i=0; i < this->bp_->size(); i++) {
        if (count_ <= 1) {
            h_buf[i] = 0.0;
        } else {
            // Population variance to copy default np.std behavior
            h_buf[i] = std::sqrt(h_buf[i] / count_);
        }
    }
}

void AvgPartialUPartialParam::avg_du_dp(double *h_buf) const {
    if (count_ > 0) {
        gpuErrchk(cudaMemcpy(h_buf, d_m_, this->bp_->size()*sizeof(*h_buf), cudaMemcpyDeviceToHost));
        return;
    }
    for(int i=0; i < this->bp_->size(); i++) {
        h_buf[i] = 0.0;
    }
}


}
