#pragma once

#include "cublas_v2.h"
#include "curand.h"

#include <vector>
#include <stdexcept>
#include <cstdio>

namespace timemachine {

// Optimizer owns *everything*. This design basically
// goes against the design of almost every text-book pattern.
// We store all memory intensive elements in here to make it easy
// to keep track of pointers and estimate total ram use.    
template<typename RealType>
class Optimizer {

public:

    virtual ~Optimizer() {};

    void step_host(
        const int num_atoms,
        const int num_dims,
        const int num_params,
        const RealType *h_dE_dx,
        const RealType *h_d2E_dx2,
        const RealType *h_d2E_dxdp, // different from device, not modified
        RealType *h_x_t, // mutable
        RealType *h_v_t, // mutable
        RealType *h_dx_dp_t, // mutable
        RealType *h_dv_dp_t, // mutable
        const RealType *d_noise_buffer
    ) const;

    virtual void step(
        const int num_atoms,
        const int num_dims,
        const int num_params,
        const RealType *dE_dx,
        const RealType *d2E_dx2,
        RealType *d2E_dxdp, // this is modified in place
        RealType *d_x_t, // mutable
        RealType *d_v_t, // mutable
        RealType *d_dx_dp_t, // mutable
        RealType *d_dv_dp_t, // mutable
        const RealType *d_noise_buffer=nullptr // optional
    ) const = 0;

};

}

