#pragma once

namespace timemachine {

template <int D>
class Gradient {

public: 

    virtual ~Gradient() {};

    /*
    Take in pointers to host memory.
    */
    void execute_host(
        const int N,
        const int P,
        const double *h_in_coords,
        const double *h_in_coords_tangents,
        const double *h_in_params,
        unsigned long long *h_out_coords,
        double *h_out_coords_tangents,
        double *h_out_params_tangents
    );

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_coords,
        const double *d_coords_tangents,
        const double *d_params,
        unsigned long long *d_out_coords,
        double *d_out_coords_tangents,
        double *d_out_params_tangents
    ) = 0;

};

}
