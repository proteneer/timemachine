#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class HarmonicBond : public Gradient<D> {

private:

    int *d_bond_idxs_;
    int *d_param_idxs_;
    int *d_lambda_idxs_;

    const int B_;

public:

    HarmonicBond(
        const std::vector<int> &bond_idxs, // [b, 2]
        const std::vector<int> &param_idxs,
        const std::vector<int> &lambda_idxs);

    ~HarmonicBond();

    virtual void execute_lambda_device(
        const int N,
        const int P,
        const double *d_coords,
        const double *d_coords_tangents,
        const double *d_params,
        const double lambda,
        const double lambda_tangents,
        unsigned long long *d_out_coords,
        double *d_out_du_dl,
        double *d_out_coords_tangents,
        double *d_out_params_tangents,
        cudaStream_t stream
    ) override;


};


}