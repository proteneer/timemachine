#pragma once

#include <memory>
#include <vector>

#include "device_buffer.hpp"
#include "potential.hpp"

namespace timemachine {

// a potential bounded to a set of parameters with some shape
struct BoundPotential {

    BoundPotential(
        // Potential *potential,
        std::shared_ptr<Potential> potential,
        std::vector<int> shape,
        const double *h_p);

    std::vector<int> shape;
    std::unique_ptr<DeviceBuffer<double>> d_p;
    std::shared_ptr<Potential> potential;

    int size() const;

    void set_params_device(const std::vector<int> shape, const double *d_p, const cudaStream_t stream);

    void execute_host(
        const int N, const double *h_x, const double *h_box, unsigned long long *h_du_dx, unsigned long long *h_u);

    void execute_device(
        const int N,
        const double *d_x,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_u,
        cudaStream_t stream) {
        this->potential->execute_device(
            N, this->size(), d_x, this->size() > 0 ? this->d_p->data : nullptr, d_box, d_du_dx, d_du_dp, d_u, stream);
    }
};

} // namespace timemachine
