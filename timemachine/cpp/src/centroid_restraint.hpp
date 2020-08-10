#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class CentroidRestraint : public Gradient {

private:

    int *d_group_a_idxs_;
    int *d_group_b_idxs_;
    double *d_masses_;

    int N_;
    int N_A_;
    int N_B_;

    double kb_;
    double b0_;

    // int *d_lambda_flags_;
    // int flat_bottom_;
    // double *d_params_;
    // double *d_du_dp_primals_;
    // double *d_du_dp_tangents_;
    // const int B_;

public:

    CentroidRestraint(
        const std::vector<int> &group_a_idxs,
        const std::vector<int> &group_b_idxs,
        const std::vector<double> &masses,
        const double kb,
        const double b0
    );

    ~CentroidRestraint();

    // int num_bonds() const {
    //     return B_;
    // }

    // void get_du_dp_primals(double *buf);

    // void get_du_dp_tangents(double *buf);

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