#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class BoreschLikeRestraint : public Gradient {

private:

    int *d_bond_idxs_;
    int *d_angle_idxs_;

    double *d_bond_params_;
    double *d_angle_params_;

    int N_A_;
    int N_B_;
    int lambda_flag_;
    int lambda_offset_;

public:

    BoreschLikeRestraint(
        const std::vector<int> &bond_idxs,
        const std::vector<int> &angle_idxs,
        const std::vector<double> &bond_params,
        const std::vector<double> &angle_params,
        const int lambda_flag,
        const int lambda_offset);

    ~BoreschLikeRestraint();

    virtual void execute_lambda_inference_device(
        const int N,
        const double *d_coords_primals,
        const double lambda_primal,
        unsigned long long *d_out_coords_primals,
        double *d_out_lambda_primals,
        double *d_out_energy_primal,
        cudaStream_t stream
    ) override;

    virtual void execute_lambda_jvp_device(
        const int N,
        const double *d_coords_primals,
        const double *d_coords_tangents,
        const double lambda_primal,
        const double lambda_tangent,
        double *d_out_coords_primals,
        double *d_out_coords_tangents,
        cudaStream_t stream
    ) override;

};


}