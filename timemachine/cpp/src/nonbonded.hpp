#pragma once

#include "neighborlist.hpp"
#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class Nonbonded : public Gradient<D> {

private:

    int *d_charge_param_idxs_;
    int *d_lj_param_idxs_;
    int *d_exclusion_idxs_; // [E,2]
    int *d_charge_scale_idxs_; // [E]
    int *d_lj_scale_idxs_; // [E]
    int *d_lambda_idxs_;

    double cutoff_;

    double *d_tmp_coords_4d_;
    double *d_tmp_grads_4d_;

    // these buffers can be in RealType as well
    // double *d_block_bounds_ctr_;
    // double *d_block_bounds_ext_;
    Neighborlist nblist_;


    const int E_;
    const int N_;


public:

    Nonbonded(
        const std::vector<int> &charge_param_idxs, // [N]
        const std::vector<int> &lj_param_idxs, // [N]
        const std::vector<int> &exclusion_idxs, // [E,2]
        const std::vector<int> &charge_scale_idxs, // [E]
        const std::vector<int> &lj_scale_idxs, // [E]
        const std::vector<int> &lambda_idxs, // [E]
        double cutoff);

    ~Nonbonded();

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