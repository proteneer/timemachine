#pragma once

#include <memory>
#include <vector>

#include "device_buffer.hpp"
#include "potential.hpp"

namespace timemachine {

// a potential bounded to a set of parameters with some shape
struct BoundPotential {

    BoundPotential(std::shared_ptr<Potential> potential, const int size, const double *h_p);

    int size;
    std::unique_ptr<DeviceBuffer<double>> d_p;
    std::shared_ptr<Potential> potential;
    const int max_size_;

    void set_params_device(const int size, const double *d_p, const cudaStream_t stream);

    void execute_host(const int N, const double *h_x, const double *h_box, unsigned long long *h_du_dx, __int128 *h_u);

    void execute_device(
        const int N,
        const double *d_x,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        __int128 *d_u,
        cudaStream_t stream) {
        this->potential->execute_device(
            N, this->size, d_x, this->size > 0 ? this->d_p->data : nullptr, d_box, d_du_dx, d_du_dp, d_u, stream);
    }
};

} // namespace timemachine
