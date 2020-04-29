#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class HarmonicBond : public Gradient {

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
    virtual void execute_lambda_device(
        const int N,
        const int P,
        const double *d_coords_primals,
        const double *d_coords_tangents,
        const double *d_params_primals,
        const double lambda_primal,
        const double lambda_tangent,
        unsigned long long *d_out_coords_primals,
        double *d_out_lambda_primals,
        double *d_energy,
        double *d_out_coords_tangents,
        double *d_out_params_tangents,
        cudaStream_t stream
    ) override;

};


}