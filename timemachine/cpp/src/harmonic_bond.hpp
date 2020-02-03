#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class HarmonicBond : public Gradient<D> {

private:

    int *d_bond_idxs_;
    int *d_param_idxs_;

    const int B_;

public:

    HarmonicBond(
        const std::vector<int> &bond_idxs, // [b, 2]
        const std::vector<int> &param_idxs);

    ~HarmonicBond();

    /*
    Execute the force computation, the semantics are:

    1. If d_coords_tangents == null, then out_coords != null, out_coords_tangent == null, out_params_tangents == null
    2. If d_coords_tangents != null, then out_coords == null, out_coords_tangent != null, out_params_tangents != null

    */
    virtual void execute_device(
        const int N,
        const int P,
        const double *d_coords,
        const double *d_coords_tangents,
        const double *d_params,
        unsigned long long *out_coords,
        double *out_coords_tangents,
        double *out_params_tangents
    ) override;


};


}