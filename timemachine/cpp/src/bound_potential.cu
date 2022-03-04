#include "bound_potential.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

BoundPotential::BoundPotential(std::shared_ptr<Potential> potential, std::vector<int> shape, const double *h_p)
    : shape(shape), d_p(this->size()), potential(potential) {
    d_p.copy_from(h_p);
}

void BoundPotential::execute_host(
    const int N,
    const double *h_x,           // [N,3]
    const double *h_box,         // [3, 3]
    const double lambda,         // [1]
    unsigned long long *h_du_dx, // [N,3]
    unsigned long long *h_du_dl, //
    unsigned long long *h_u) {

    const int D = 3;

    DeviceBuffer<double> d_x(N * D);
    DeviceBuffer<double> d_box(D * D);

    d_x.copy_from(h_x);
    d_box.copy_from(h_box);

    DeviceBuffer<unsigned long long> d_du_dx(N * D);
    DeviceBuffer<unsigned long long> d_du_dl(N);
    DeviceBuffer<unsigned long long> d_u(N);

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMemset(d_du_dx.data, 0, d_du_dx.size));
    gpuErrchk(cudaMemset(d_du_dl.data, 0, d_du_dl.size));
    gpuErrchk(cudaMemset(d_u.data, 0, d_u.size));

    this->execute_device(
        N, d_x.data, d_box.data, lambda, d_du_dx.data, nullptr, d_du_dl.data, d_u.data, static_cast<cudaStream_t>(0));

    d_du_dx.copy_to(h_du_dx);
    d_du_dl.copy_to(h_du_dl);
    d_u.copy_to(h_u);
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
