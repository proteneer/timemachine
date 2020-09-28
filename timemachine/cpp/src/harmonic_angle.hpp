#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class HarmonicAngle : public Potential {

private:

    int *d_angle_idxs_;

    // double *d_params_;
    // double *d_du_dp_primals_;
    // double *d_du_dp_tangents_;

    const int A_;

public:

    HarmonicAngle(
        const std::vector<int> &angle_idxs
    );

    ~HarmonicAngle();

    // int num_angles() const {
    //     return A_;
    // }

    // void get_du_dp_primals(double *buf);

    // void get_du_dp_tangents(double *buf);

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) override;


    // virtual void execute_lambda_inference_device(
    //     const int N,
    //     const double *d_coords_primals,
    //     const double lambda_primal,
    //     unsigned long long *d_out_coords_primals,
    //     double *d_out_lambda_primals,
    //     double *d_out_energy_primal,
    //     cudaStream_t stream
    // ) override;

    // virtual void execute_lambda_jvp_device(
    //     const int N,
    //     const double *d_coords_primals,
    //     const double *d_coords_tangents,
    //     const double lambda_primal,
    //     const double lambda_tangent,
    //     double *d_out_coords_primals,
    //     double *d_out_coords_tangents,
    //     cudaStream_t stream
    // ) override;


};


}