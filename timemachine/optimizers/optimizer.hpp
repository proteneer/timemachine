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
template<typename NumericType>
class Optimizer {

public:

    virtual void opt_init(
        const int num_atoms,
        const NumericType *h_x0,
        const NumericType *h_v0) = 0;

    virtual void apply_fn(
        const NumericType *d_dE_dx,
        const NumericType *d_full_d2E_dxdp) = 0;


};

}

