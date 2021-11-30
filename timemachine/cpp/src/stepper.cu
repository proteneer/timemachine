#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "stepper.hpp"

#define PI 3.1415926535897932384626433

#include <iostream>
namespace timemachine {

Stepper::Stepper(int F) : streams_(F) {
    for (int i = 0; i < F; i++) {
        gpuErrchk(cudaStreamCreate(&streams_[i]));
    }
}

Stepper::~Stepper() {
    for (int i = 0; i < streams_.size(); i++) {
        gpuErrchk(cudaStreamDestroy(streams_[i]));
    }
}

cudaStream_t Stepper::get_stream(int idx) { return streams_[idx]; }

void Stepper::sync_all_streams() {
    for (int i = 0; i < streams_.size(); i++) {
        gpuErrchk(cudaStreamSynchronize(streams_[i]));
    }
}

AlchemicalStepper::~AlchemicalStepper() {
    gpuErrchk(cudaFree(d_du_dl_));
    gpuErrchk(cudaFree(d_energies_));
}

AlchemicalStepper::AlchemicalStepper(std::vector<Gradient *> forces, const std::vector<double> &lambda_schedule)
    : forces_(forces), lambda_schedule_(lambda_schedule), count_(0), Stepper(forces.size()) {

    const int F = forces.size();
    const int T = lambda_schedule_.size();

    gpuErrchk(cudaMalloc(&d_du_dl_, F * T * sizeof(*d_du_dl_)));
    gpuErrchk(cudaMemset(d_du_dl_, 0, F * T * sizeof(*d_du_dl_)));

    gpuErrchk(cudaMalloc(&d_energies_, T * sizeof(*d_energies_)));
    gpuErrchk(cudaMemset(d_energies_, 0, T * sizeof(*d_energies_)));
}

void AlchemicalStepper::forward_step(const int N, const double *coords, unsigned long long *dx) {

    const int T = lambda_schedule_.size();

    gpuErrchk(cudaDeviceSynchronize());
    for (int f = 0; f < forces_.size(); f++) {
        forces_[f]->execute_lambda_inference_device(
            N,
            coords,
            lambda_schedule_[count_],
            dx,                        // forces
            d_du_dl_ + f * T + count_, // du_dl
            d_energies_ + count_,      // energies
            this->get_stream(f));
    }

    gpuErrchk(cudaDeviceSynchronize());
    count_ += 1;
};

void AlchemicalStepper::backward_step(
    const int N,
    const double *coords,
    const double *dx_tangent,
    double *coords_jvp_primals,
    double *coords_jvp_tangents) {

    count_ -= 1;

    gpuErrchk(cudaDeviceSynchronize());
    if (count_ >= du_dl_adjoint_.size()) {
        throw std::runtime_error("You probably forgot to set du_dl adjoints!\n");
    }

    const int T = lambda_schedule_.size();

    for (int f = 0; f < forces_.size(); f++) {
        forces_[f]->execute_lambda_jvp_device(
            N,
            coords,
            dx_tangent,
            lambda_schedule_[count_],
            du_dl_adjoint_[f * T + count_], // FIX
            coords_jvp_primals,
            coords_jvp_tangents,
            this->get_stream(f));
    }
    gpuErrchk(cudaDeviceSynchronize());
};

void AlchemicalStepper::get_du_dl(double *buf) {
    const int T = get_T();
    const int F = forces_.size();
    cudaMemcpy(buf, d_du_dl_, T * F * sizeof(double), cudaMemcpyDeviceToHost);
};

void AlchemicalStepper::get_energies(double *buf) {
    const int T = get_T();
    cudaMemcpy(buf, d_energies_, T * sizeof(double), cudaMemcpyDeviceToHost);
};

void AlchemicalStepper::set_du_dl_adjoint(const int FT, const double *adj) {
    if (FT != lambda_schedule_.size() * get_F()) {
        throw std::runtime_error("adjoint size not the same as lambda schedule size");
    }
    du_dl_adjoint_.resize(FT);
    memcpy(&du_dl_adjoint_[0], adj, FT * sizeof(double));
};

}; // namespace timemachine
