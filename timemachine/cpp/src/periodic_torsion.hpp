#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class PeriodicTorsion : public Gradient<D> {

private:

    int *d_torsion_idxs_;
    int *d_param_idxs_;

    const int T_;

public:

    PeriodicTorsion(
        const std::vector<int> &torsion_idxs, // [n, 4]
        const std::vector<int> &param_idxs);

    ~PeriodicTorsion();

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
        double *out_params_tangents,
        cudaStream_t stream
    ) override;


};


}