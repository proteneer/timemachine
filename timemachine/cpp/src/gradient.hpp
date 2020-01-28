#pragma once

namespace timemachine {

template <typename RealType, int D>
class Gradient {

public: 

    virtual ~Gradient() {};

    /*
    Take in pointers to host memory.
    */
    void execute_host(
        const int N,
        const int P,
        const RealType *h_in_coords,
        const RealType *h_in_coords_tangents,
        const RealType *h_in_params,
        unsigned long long *h_out_coords,
        RealType *h_out_coords_tangents,
        RealType *h_out_params_tangents
    );

    virtual void execute_device(
        const int N,
        const int P,
        const RealType *d_coords,
        const RealType *d_coords_tangents,
        const RealType *d_params,
        unsigned long long *d_out_coords,
        RealType *d_out_coords_tangents,
        RealType *d_out_params_tangents
    ) = 0;

};

}
