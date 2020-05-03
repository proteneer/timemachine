#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class HarmonicBond : public Gradient {

private:

    int *d_bond_idxs_;
    int *d_k_idxs_;
    int *d_b_idxs_;
    int *d_k_idxs_pi_;
    int *d_b_idxs_pi_;

    const int B_;

public:

    HarmonicBond(
        const std::vector<int> &bond_idxs, // [b, 2]
        const std::vector<int> &k_idxs,
        const std::vector<int> &b_idxs,
        const std::vector<int> &k_idxs_pi,
        const std::vector<int> &b_idxs_pi);

    ~HarmonicBond();

    virtual void execute_lambda_inference_device(
        const int N,
        const int P,
        const double *d_coords_primals,
        const double *d_params_primals,
        const double lambda_primal,
        unsigned long long *d_out_coords_primals,
        double *d_out_lambda_primals,
        double *d_out_energy_primal,
        cudaStream_t stream
    ) override;


    virtual void execute_lambda_jvp_device(
        const int N,
        const int P,
        const double *d_coords_primals,
        const double *d_coords_tangents,
        const double *d_params_primals,
        const double lambda_primal,
        const double lambda_tangent,
        double *d_out_coords_primals,
        double *d_out_coords_tangents,
        double *d_out_params_primals,
        double *d_out_params_tangents,
        cudaStream_t stream
    ) override;

};


}