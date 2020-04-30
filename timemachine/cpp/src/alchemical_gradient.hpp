#pragma once

#include "gradient.hpp"

namespace timemachine {

class AlchemicalGradient : public Gradient {

private:

    Gradient* u0_;
    Gradient* u1_;

    // TDB we can get away with less buffer use
    unsigned long long *d_out_coords_primals_buffer_u0_;
    double *d_out_energy_primal_buffer_u0_;
    double *d_out_lambda_primal_buffer_u0_;

    double *d_out_jvp_coords_primals_buffer_u0_;
    double *d_out_jvp_coords_tangents_buffer_u0_;
    double *d_out_jvp_params_primals_buffer_u0_;
    double *d_out_jvp_params_tangents_buffer_u0_;

    unsigned long long *d_out_coords_primals_buffer_u1_;
    double *d_out_energy_primal_buffer_u1_;
    double *d_out_lambda_primal_buffer_u1_;

    double *d_out_jvp_coords_primals_buffer_u1_;
    double *d_out_jvp_coords_tangents_buffer_u1_;
    double *d_out_jvp_params_primals_buffer_u1_;
    double *d_out_jvp_params_tangents_buffer_u1_;

public: 

    virtual ~AlchemicalGradient();

    AlchemicalGradient(
        const int N,
        const int P,
        Gradient* u0,
        Gradient* u1);

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
