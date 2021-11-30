#include "bound_potential.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

BoundPotential::BoundPotential(std::shared_ptr<Potential> potential, std::vector<int> shape, const double *h_p)
    : potential(potential), shape(shape) {

    int P = this->size();

    gpuErrchk(cudaMalloc(&d_p, P * sizeof(*d_p)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P * sizeof(*d_p), cudaMemcpyHostToDevice));
}

BoundPotential::~BoundPotential() {
    // only free the d_ps, but not the pure potentials themselves
    gpuErrchk(cudaFree(d_p));
}

void BoundPotential::execute_host(
    const int N,
    const double *h_x,           // [N,3]
    const double *h_box,         // [3, 3]
    const double lambda,         // [1]
    unsigned long long *h_du_dx, // [N,3]
    unsigned long long *h_du_dl, //
    unsigned long long *h_u) {

    double *d_x;
    double *d_box;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_x, N * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N * D * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D * D * sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx; // du/dx
    unsigned long long *d_du_dl; // du/dl
    unsigned long long *d_u;     // u

    const int P = this->size();

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_du_dx, N * D * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dx, 0, N * D * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_du_dl, N * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dl, 0, N * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_u, N * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_u, 0, N * sizeof(unsigned long long)));

    this->execute_device(N, d_x, d_box, lambda, d_du_dx, nullptr, d_du_dl, d_u, static_cast<cudaStream_t>(0));

    gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N * D * sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dx));
    gpuErrchk(cudaMemcpy(h_du_dl, d_du_dl, N * sizeof(*h_du_dl), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dl));
    gpuErrchk(cudaMemcpy(h_u, d_u, N * sizeof(*h_u), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_box));
};

int BoundPotential::size() const {
    if (shape.size() == 0) {
        return 0;
    }
    int total = 1;
    for (auto s : shape) {
        total *= s;
    }
    return total;
}

} // namespace timemachine
