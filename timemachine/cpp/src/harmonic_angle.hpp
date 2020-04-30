#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class HarmonicAngle : public Gradient {

private:

    int *d_angle_idxs_;
    int *d_param_idxs_;

    const int A_;

public:

    HarmonicAngle(
        const std::vector<int> &angle_idxs, // [A, 3]
        const std::vector<int> &param_idxs); // [A, 2]

    ~HarmonicAngle();

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