#pragma once

#include <vector>

#include "potential.hpp"

namespace timemachine {

// a potential bounded to a set of parameters with some shape
struct BoundPotential {

    BoundPotential(
        std::vector<int> shape,
        double *h_p
    );

    ~BoundPotential();

    double *d_p;
    Potential *potential;
    std::vector<int> shape;

    int size() const;

    void execute_device(
        const int N,
        const double *d_x,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream) {
        this->potential->execute_device(
            N,
            this->size(),
            d_x,
            this->d_p,
            d_box,
            lambda,
            d_du_dx,
            d_du_dp,
            d_du_dl,
            d_u,
            stream
        );
    }

};


} // namespace timemachine