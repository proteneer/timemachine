#include "context.hpp"
#include "gpu_utils.cuh"
#include <iostream>

namespace timemachine {

Context::Context(
    int N,
    const double *x_0,
    const double *v_0,
    const double *box_0,
    // double lambda,
    Integrator* intg,
    std::vector<BoundPotential *> bps) :
    // std::vector<Observable *> obs) : 
    N_(N),
    intg_(intg),
    bps_(bps),
    // observables_(obs),
    // lambda_(lambda),
    step_(0) {

    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N*3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N*3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3*3);

    gpuErrchk(cudaMalloc(&d_du_dx_t_, N*3*sizeof(*d_du_dx_t_)));
    gpuErrchk(cudaMalloc(&d_u_t_, 1*sizeof(*d_u_t_)));

};

Context::~Context() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_box_t_));
    gpuErrchk(cudaFree(d_u_t_));
    gpuErrchk(cudaFree(d_du_dx_t_));
};

void Context::add_observable(Observable *obs) {
    this->observables_.push_back(obs);
}

void Context::step(double lambda) {

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
    gpuErrchk(cudaMemset(d_u_t_, 0, 1*sizeof(*d_du_dx_t_)));

    for(int i=0; i < bps_.size(); i++) {
        // std::cout << i << std::endl;
        bps_[i]->execute_device(
            N_,
            d_x_t_,
            d_box_t_,
            lambda,
            d_du_dx_t_,
            nullptr,
            nullptr,
            d_u_t_,
            static_cast<cudaStream_t>(0) // TBD: parallelize me!
        );
    }

    intg_->step_fwd(
        d_x_t_,
        d_v_t_,
        d_du_dx_t_,
        d_box_t_
    );

    cudaDeviceSynchronize();

    step_ += 1;

};


int Context::num_atoms() const {
    return N_;
}

double Context::get_u_t() const {
    double u;
    gpuErrchk(cudaMemcpy(&u, d_u_t_, 1*sizeof(*d_u_t_), cudaMemcpyDeviceToHost));
    return u;
}

void Context::get_du_dx_t(unsigned long long *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_du_dx_t_, N_*3*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_x_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_x_t_, N_*3*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_v_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_v_t_, N_*3*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}


// void Context::compute(unsigned int flags) {

//     double *u = (flags & ComputeFlags::u) ? d_u_t_ : nullptr;
//     unsigned long long *du_dx = (flags & ComputeFlags::du_dx) ? d_du_dx_t_ : nullptr;
//     double *du_dl = (flags & ComputeFlags::du_dl) ? d_du_dl_t_ : nullptr;

//     for(int i=0; i < potentials_.size(); i++) {

//         DualParams *dp = dual_params_[i];

//         // note that dp->d_du_dp itself may be null if the end-user
//         // does not care about du_dp.
//         double *du_dp = (flags & ComputeFlags::du_dp) ? dp->d_du_dp : nullptr;

//         potentials_[i]->execute_device(
//             N_,
//             dp->size(),
//             d_x_t_,
//             dp->d_p,
//             d_box_t_,
//             lambda_,
//             du_dx,
//             du_dp,
//             du_dl,
//             u,
//             static_cast<cudaStream_t>(0)
//         );

//     }

// };


}