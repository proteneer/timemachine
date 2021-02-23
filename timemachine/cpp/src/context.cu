#include "context.hpp"
#include "gpu_utils.cuh"
#include "fixed_point.hpp"
#include <iostream>
#include <chrono>
#include <cub/cub.cuh>

namespace timemachine {

Context::Context(
    int N,
    const double *x_0,
    const double *v_0,
    const double *box_0,
    Integrator* intg,
    std::vector<BoundPotential *> bps) :
    N_(N),
    intg_(intg),
    bps_(bps),
    step_(0),
    d_sum_storage_(nullptr),
    d_sum_storage_bytes_(0) {

    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N*3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N*3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3*3);

    gpuErrchk(cudaMalloc(&d_du_dx_t_, N*3*sizeof(*d_du_dx_t_)));
    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, N*sizeof(*d_du_dl_buffer_)));

    unsigned long long *d_in_tmp_ = nullptr; // dummy
    unsigned long long *d_out_tmp_ = nullptr; // dummy

    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_in_tmp_, d_out_tmp_, N_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sum_storage_, d_sum_storage_bytes_));

    // for(int i=0; i < bps.size(); i++) {
        // cudaStream_t stream;
        // gpuErrchk(cudaStreamCreate(&stream));
        // streams_.push_back(stream);
    // }


};

Context::~Context() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_box_t_));
    gpuErrchk(cudaFree(d_du_dx_t_));
    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_sum_storage_));

    // for(int i=0; i < streams_.size(); i++) {
        // gpuErrchk(cudaStreamDestroy(streams_[i]));
    // }
};

void Context::add_observable(Observable *obs) {
    this->observables_.push_back(obs);
}

std::vector<double> Context::multiple_steps(
    const std::vector<double> &lambda_schedule,
    int store_du_dl_freq) {
    unsigned long long *d_du_dl_buffer = nullptr;
    // try catch block is to deal with leaks in d_du_dl_buffer
    if(store_du_dl_freq==0) {
        store_du_dl_freq = lambda_schedule.size();
    }
    int buffer_size = (lambda_schedule.size() + store_du_dl_freq - 1) / store_du_dl_freq;
    try {
        // indicator so we can set it to a default arg.
        gpuErrchk(cudaMalloc(&d_du_dl_buffer, buffer_size*sizeof(*d_du_dl_buffer)));
        gpuErrchk(cudaMemset(d_du_dl_buffer, 0, buffer_size*sizeof(*d_du_dl_buffer)));
        for(int i=0; i < lambda_schedule.size(); i++) {
            // decide if we need to store the du_dl for this step
            unsigned long long *du_dl_ptr = nullptr;
            if(i % store_du_dl_freq == 0) {
                // pemdas but just to make it clear we're doing pointer arithmetic
                du_dl_ptr = d_du_dl_buffer + (i / store_du_dl_freq);
            }
            double lambda = lambda_schedule[i];
            this->_step(lambda, du_dl_ptr);
        }
        cudaDeviceSynchronize();

        std::vector<unsigned long long> h_du_dl_buffer_ull(buffer_size);
        gpuErrchk(cudaMemcpy(
            &h_du_dl_buffer_ull[0],
            d_du_dl_buffer,
            buffer_size*sizeof(*d_du_dl_buffer),
            cudaMemcpyDeviceToHost)
        );

        std::vector<double> h_du_dl_buffer_double(buffer_size);
        for(int i=0; i < h_du_dl_buffer_ull.size(); i++) {
            h_du_dl_buffer_double[i] = FIXED_TO_FLOAT<double>(h_du_dl_buffer_ull[i]);
        }

        gpuErrchk(cudaFree(d_du_dl_buffer));
        return h_du_dl_buffer_double;

    } catch(...) {
        gpuErrchk(cudaFree(d_du_dl_buffer));
        throw;
    }

}

void Context::step(double lambda) {
    this->_step(lambda, nullptr);
    cudaDeviceSynchronize();
}

void Context::_step(double lambda, unsigned long long *du_dl_out) {

    // the observables decide on whether or not to act on given
    // data (cheap pointers in any case)

    for(int i=0; i < observables_.size(); i++) {
        observables_[i]->observe(
            step_,
            N_,
            d_x_t_,
            d_box_t_,
            lambda
        );
    }

    gpuErrchk(cudaMemset(d_du_dx_t_, 0, N_*3*sizeof(*d_du_dx_t_)));

    if(du_dl_out) {
        gpuErrchk(cudaMemset(d_du_dl_buffer_, 0, N_*sizeof(*d_du_dl_buffer_)));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i < bps_.size(); i++) {

        bps_[i]->execute_device(
            N_,
            d_x_t_,
            d_box_t_,
            lambda,
            d_du_dx_t_, // we only need the forces
            nullptr,
            du_dl_out ? d_du_dl_buffer_ : nullptr,
            nullptr,
            static_cast<cudaStream_t>(0) // TBD: parallelize me!
            // streams_[i]
        );
    }

    // compute du_dl
    if(du_dl_out) {
        cub::DeviceReduce::Sum(
            d_sum_storage_,
            d_sum_storage_bytes_,
            d_du_dl_buffer_,
            du_dl_out,
            N_,
            static_cast<cudaStream_t>(0)
        );
        gpuErrchk(cudaPeekAtLastError());
    }


    // for(int i=0; i < streams_.size(); i++) {
        // gpuErrchk(cudaStreamSynchronize(streams_[i]));
    // }

    intg_->step_fwd(
        d_x_t_,
        d_v_t_,
        d_du_dx_t_,
        d_box_t_
    );

    step_ += 1;

};


int Context::num_atoms() const {
    return N_;
}

void Context::get_du_dx_t_minus_1(unsigned long long *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_du_dx_t_, N_*3*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_x_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_x_t_, N_*3*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_v_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_v_t_, N_*3*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

}