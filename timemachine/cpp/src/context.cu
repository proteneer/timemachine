#include "context.hpp"
#include "gpu_utils.cuh"
#include <iostream>
#include <chrono>

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
    step_(0) {

    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N*3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N*3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3*3);

    gpuErrchk(cudaMalloc(&d_du_dx_t_, N*3*sizeof(*d_du_dx_t_)));

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

    // for(int i=0; i < streams_.size(); i++) {
        // gpuErrchk(cudaStreamDestroy(streams_[i]));
    // }
};

void Context::add_observable(Observable *obs) {
    this->observables_.push_back(obs);
}

void Context::multiple_steps(std::vector<double> lambda_schedule) {
    for(auto lambda : lambda_schedule) {
        this->_step(lambda);
    }
    cudaDeviceSynchronize();
}

void Context::step(double lambda) {
    this->_step(lambda);
    cudaDeviceSynchronize();
}

void Context::_step(double lambda) {

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

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i < bps_.size(); i++) {

        // std::cout << i << std::endl;

        bps_[i]->execute_device(
            N_,
            d_x_t_,
            d_box_t_,
            lambda,
            d_du_dx_t_, // we only need the forces
            nullptr,
            nullptr,
            nullptr,
            static_cast<cudaStream_t>(0) // TBD: parallelize me!
            // streams_[i]
        );
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

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