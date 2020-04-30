#pragma once

#include "neighborlist.hpp"
#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class Nonbonded : public Gradient {

private:

    int *d_charge_param_idxs_;
    int *d_lj_param_idxs_;
    int *d_exclusion_idxs_; // [E,2]
    int *d_charge_scale_idxs_; // [E]
    int *d_lj_scale_idxs_; // [E]
    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;

    double cutoff_;
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
        const std::vector<int> &lambda_plane_idxs, // N
        const std::vector<int> &lambda_offset_idxs, // N
        double cutoff);

    ~Nonbonded();

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