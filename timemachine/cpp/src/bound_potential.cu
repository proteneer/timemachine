#include "bound_potential.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

BoundPotential::BoundPotential(std::shared_ptr<Potential> potential, const std::vector<double> &params)
    : size(params.size()), buffer_size_(size), d_p(buffer_size_), potential(potential) {
    set_params(params);
}

void BoundPotential::execute_device(
    const int N,
    const double *d_x,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {
    this->potential->execute_device(
        N, this->size, d_x, this->size > 0 ? this->d_p.data : nullptr, d_box, d_du_dx, d_du_dp, d_u, stream);
}

void BoundPotential::execute_host(
    const int N,
    const double *h_x,           // [N,3]
    const double *h_box,         // [3, 3]
    unsigned long long *h_du_dx, // [N, 3]
    __int128 *h_u                // [1]
) {

    const int D = 3;

    DeviceBuffer<double> d_x(N * D);
    DeviceBuffer<double> d_box(D * D);

    d_x.copy_from(h_x);
    d_box.copy_from(h_box);

    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dx(nullptr);
    std::unique_ptr<DeviceBuffer<__int128>> d_u(nullptr);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    // very important that these are initialized to zero since the kernels themselves just accumulate
    if (h_du_dx != nullptr) {
        d_du_dx.reset(new DeviceBuffer<unsigned long long>(N * D));
        gpuErrchk(cudaMemsetAsync(d_du_dx->data, 0, d_du_dx->size, stream));
    }
    if (h_u != nullptr) {
        d_u.reset(new DeviceBuffer<__int128>(1));
        gpuErrchk(cudaMemsetAsync(d_u->data, 0, d_u->size, stream));
    }

    this->execute_device(
        N,
        d_x.data,
        d_box.data,
        h_du_dx != nullptr ? d_du_dx->data : nullptr,
        nullptr,
        h_u != nullptr ? d_u->data : nullptr,
        stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    if (h_du_dx) {
        d_du_dx->copy_to(h_du_dx);
    }
    if (h_u) {
        d_u->copy_to(h_u);
    }
};

void BoundPotential::set_params(const std::vector<double> &params) {
    if (params.size() != buffer_size_) {
        throw std::runtime_error(
            "parameter size is not equal to device buffer size: " + std::to_string(params.size()) +
            " != " + std::to_string(buffer_size_));
    }
    d_p.copy_from(params.data());
    this->size = params.size();
}

void BoundPotential::set_params_device(const int new_size, const double *d_new_params, const cudaStream_t stream) {
    if (static_cast<size_t>(new_size) > buffer_size_) {
        throw std::runtime_error(
            "parameter size is greater than device buffer size: " + std::to_string(new_size) + " > " +
            std::to_string(buffer_size_));
    }
    gpuErrchk(cudaMemcpyAsync(d_p.data, d_new_params, new_size * sizeof(*d_p.data), cudaMemcpyDeviceToDevice, stream));
    this->size = new_size;
}
} // namespace timemachine
