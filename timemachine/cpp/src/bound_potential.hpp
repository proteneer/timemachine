#pragma once

#include <memory>
#include <vector>

#include "device_buffer.hpp"
#include "potential.hpp"
#include "types.hpp"

namespace timemachine {

// a potential bounded to a set of parameters with some shape
struct BoundPotential {

    BoundPotential(std::shared_ptr<Potential> potential, const std::vector<ParamsType> &params);

    int size;
    const size_t buffer_size_; // d_p.size / sizeof(*d_p.data) TODO: remove when DeviceBuffer::length or similar added
    DeviceBuffer<ParamsType> d_p;
    std::shared_ptr<Potential> potential;

    void set_params(const std::vector<ParamsType> &params);

    void set_params_device(const int size, const ParamsType *d_p, const cudaStream_t stream);

    void
    execute_host(const int N, const double *h_x, const double *h_box, unsigned long long *h_du_dx, EnergyType *h_u);

    void execute_device(
        const int N,
        const double *d_x,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        EnergyType *d_u,
        cudaStream_t stream);
};

} // namespace timemachine
