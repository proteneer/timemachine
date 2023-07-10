#include "bound_potential.hpp"
#include "gpu_utils.cuh"

int compute_size(std::vector<int> shape) {
    if (shape.size() == 0) {
        return 0;
    }
    int total = 1;
    for (auto s : shape) {
        total *= s;
    }
    return total;
}

namespace timemachine {

BoundPotential::BoundPotential(std::shared_ptr<Potential> potential, std::vector<int> shape, const double *h_p)
    : shape(shape), d_p(nullptr), potential(potential), max_size_(compute_size(shape)) {
    if (this->size() > 0) {
        d_p.reset(new DeviceBuffer<double>(this->size()));
        d_p->copy_from(h_p);
    }
}

void BoundPotential::execute_host(
    const int N,
    const double *h_x,           // [N,3]
    const double *h_box,         // [3, 3]
    unsigned long long *h_du_dx, // [N, 3]
    unsigned long long *h_u,     // [1]
    int *h_u_overflow_count      // [1]
) {

    const int D = 3;

    DeviceBuffer<double> d_x(N * D);
    DeviceBuffer<double> d_box(D * D);

    d_x.copy_from(h_x);
    d_box.copy_from(h_box);

    DeviceBuffer<unsigned long long> d_du_dx(N * D);
    DeviceBuffer<unsigned long long> d_u(1);
    DeviceBuffer<int> d_u_overflow_count(1);

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMemset(d_du_dx.data, 0, d_du_dx.size));
    gpuErrchk(cudaMemset(d_u.data, 0, d_u.size));
    gpuErrchk(cudaMemset(d_u_overflow_count.data, 0, d_u_overflow_count.size));

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->execute_device(N, d_x.data, d_box.data, d_du_dx.data, nullptr, d_u.data, d_u_overflow_count.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    if (h_du_dx) {
        d_du_dx.copy_to(h_du_dx);
    }
    if (h_u) {
        d_u.copy_to(h_u);
        d_u_overflow_count.copy_to(h_u_overflow_count);
    }
};

void BoundPotential::set_params_device(
    const std::vector<int> device_shape, const double *d_new_params, const cudaStream_t stream) {
    int updated_size = compute_size(device_shape);
    if (updated_size > 0) {
        if (updated_size > max_size_) {
            throw std::runtime_error(
                "parameter size is greater than max size: " + std::to_string(updated_size) + " > " +
                std::to_string(max_size_));
        }
        gpuErrchk(cudaMemcpyAsync(
            d_p->data, d_new_params, updated_size * sizeof(*d_p->data), cudaMemcpyDeviceToDevice, stream));
    }
    shape = device_shape;
}

int BoundPotential::size() const { return compute_size(this->shape); }

} // namespace timemachine
