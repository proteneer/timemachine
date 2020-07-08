#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class PeriodicTorsion : public Gradient {

private:

    int *d_torsion_idxs_;

    double *d_params_;
    double *d_du_dp_primals_;
    double *d_du_dp_tangents_;

    const int T_;

public:

    PeriodicTorsion(
        const std::vector<int> &torsion_idxs, // [n, 4]
        const std::vector<double> &params
    );

    ~PeriodicTorsion();

    int num_torsions() const {
        return T_;
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