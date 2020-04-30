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


AlchemicalStepper::~AlchemicalStepper() {
    gpuErrchk(cudaFree(d_du_dl_));
    gpuErrchk(cudaFree(d_energies_));
}

AlchemicalStepper::AlchemicalStepper(
    std::vector<Gradient*> forces,
    const std::vector<double> &lambda_schedule
) : forces_(forces),
    lambda_schedule_(lambda_schedule),
    count_(0),
    Stepper(forces.size()) {

    const int T = lambda_schedule_.size();

    gpuErrchk(cudaMalloc(&d_du_dl_, T*sizeof(*d_du_dl_)));
    gpuErrchk(cudaMemset(d_du_dl_, 0, T*sizeof(*d_du_dl_)));

    gpuErrchk(cudaMalloc(&d_energies_, T*sizeof(*d_energies_)));
    gpuErrchk(cudaMemset(d_energies_, 0, T*sizeof(*d_energies_)));

}

void AlchemicalStepper::forward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    unsigned long long *dx) {

    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_lambda_inference_device(
            N,
            P,
            coords,
            params,
            lambda_schedule_[count_],
            dx, // forces
            d_du_dl_ + count_, // du_dl
            d_energies_ + count_, // energies
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
    double *coords_jvp_primals,
    double *coords_jvp_tangents,
    double *params_jvp_primals,
    double *params_jvp_tangents) {

    count_ -= 1;

    gpuErrchk(cudaDeviceSynchronize());
    if(count_ >= du_dl_adjoint_.size()) {
        throw std::runtime_error("You probably forgot to set du_dl adjoints!\n");
    }
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_lambda_jvp_device(
            N,
            P,
            coords,
            dx_tangent,
            params,
            lambda_schedule_[count_],
            du_dl_adjoint_[count_],
            coords_jvp_primals,
            coords_jvp_tangents,
            params_jvp_primals,
            params_jvp_tangents,
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
