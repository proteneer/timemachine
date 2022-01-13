#pragma once

#include <memory>
#include <vector>

#include "potential.hpp"

namespace timemachine {

// a potential bounded to a set of parameters with some shape
struct BoundPotential {

    BoundPotential(
        // Potential *potential,
        std::shared_ptr<Potential> potential,
        std::vector<int> shape,
        const double *h_p);

    ~BoundPotential();

    double *d_p;
    std::shared_ptr<Potential> potential;
    std::vector<int> shape;

    int size() const;

    void execute_host(
        const int N,
        const double *h_x,
        const double *h_box,
        const double lambda, // lambda
        unsigned long long *h_du_dx,
        unsigned long long *h_du_dl,
        unsigned long long *h_u);

    void execute_device(
        const int N,
        const double *d_x,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_du_dl,
        unsigned long long *d_u,
        cudaStream_t stream) {
        this->potential->execute_device(
            N, this->size(), d_x, this->d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
    }
};

} // namespace timemachine
