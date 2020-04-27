#include "stepper.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"

#define PI  3.1415926535897932384626433

#include <iostream>
namespace timemachine {

Stepper::Stepper(int F) : streams_(F) {
    for(int i=0; i < F; i++) {
        gpuErrchk(cudaStreamCreate(&streams_[i]));        
    }

}

Stepper::~Stepper() {
    for(int i=0; i <streams_.size(); i++) {
        gpuErrchk(cudaStreamDestroy(streams_[i]));
    }
}

cudaStream_t Stepper::get_stream(int idx) {
    return streams_[idx];
}

void Stepper::sync_all_streams() {
    for(int i=0; i < streams_.size(); i++) {
        gpuErrchk(cudaStreamSynchronize(streams_[i]));
    }
}

AlchemicalStepper::AlchemicalStepper(
    std::vector<Gradient<3> *> forces,
    const std::vector<double> &lambda_schedule
) : forces_(forces),
    lambda_schedule_(lambda_schedule),
    count_(0),
    Stepper(forces.size()) {

    const int T = lambda_schedule_.size();

    gpuErrchk(cudaMalloc(&d_du_dl_, T*sizeof(*d_du_dl_)));
    gpuErrchk(cudaMemset(d_du_dl_, 0, T*sizeof(*d_du_dl_)));

}

void AlchemicalStepper::forward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    unsigned long long *dx) {

    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_lambda_device(
            N,
            P,
            coords,
            nullptr,
            params,
            lambda_schedule_[count_],
            0,
            dx, // forces
            d_du_dl_ + count_, // du_dl
            nullptr,
            nullptr,
            this->get_stream(f)
        );
    }

    gpuErrchk(cudaDeviceSynchronize());
    count_ += 1;

};

void AlchemicalStepper::backward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    const double *dx_tangent,
    double *coords_jvp,
    double *params_jvp) {

    count_ -= 1;

    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_lambda_device(
            N,
            P,
            coords,
            dx_tangent,
            params,
            lambda_schedule_[count_],
            du_dl_adjoint_[count_],
            nullptr,
            nullptr,
            coords_jvp,
            params_jvp,
            this->get_stream(f)
        );
    }
    gpuErrchk(cudaDeviceSynchronize());


};

void AlchemicalStepper::get_du_dl(
    double *buf) {
    const int T = get_T();
    cudaMemcpy(buf, d_du_dl_, T*sizeof(double), cudaMemcpyDeviceToHost);
};

void AlchemicalStepper::set_du_dl_adjoint(
    const int T,
    const double *adj) {
    if(T != lambda_schedule_.size()) {
        throw std::runtime_error("adjoint size not the same as lambda schedule size");
    }
    du_dl_adjoint_.resize(T);
    memcpy(&du_dl_adjoint_[0], adj, T*sizeof(double));
};

};
