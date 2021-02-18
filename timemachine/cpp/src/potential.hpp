#pragma once

#include <cuda_runtime.h>

namespace timemachine {

// *Not* guaranteed to be thread-safe.
class Potential {

public:

    virtual ~Potential() {};

    void execute_host(
        const int N,
        const int P,
        const double *h_x,
        const double *h_p,
        const double *h_box,
        const double lambda, // lambda
        unsigned long long *h_du_dx,
        double *h_du_dp,
        double *h_du_dl,
        double *h_u
    );

    void execute_host_du_dx(
        const int N,
        const int P,
        const double *h_x,
        const double *h_p,
        const double *h_box,
        const double lambda, // lambda
        unsigned long long *h_du_dx
    );

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) = 0;

};

}
