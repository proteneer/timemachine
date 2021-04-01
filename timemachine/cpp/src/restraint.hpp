#pragma once

#include "gradient.hpp"
#include <vector>

namespace jankmachine {

template<typename RealType>
class Restraint : public Gradient {

private:

    int *d_bond_idxs_;

    int *d_lambda_flags_;
    int flat_bottom_;

    double *d_params_;
    double *d_du_dp_primals_;
    double *d_du_dp_tangents_;

    const int B_;

public:

    Restraint(
        const std::vector<int> &bond_idxs, // [b, 2]
        const std::vector<double> &params,
        const std::vector<int> &lambda_flags);

    ~Restraint();

    int num_bonds() const {
        return B_;
    }


    void get_du_dp_primals(double *buf);

    void get_du_dp_tangents(double *buf);

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