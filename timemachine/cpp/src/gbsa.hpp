#pragma once

#include "neighborlist.hpp"
#include "gradient.hpp"
#include "surreal.cuh"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class GBSA : public Gradient<D> {

private:

    int *d_charge_param_idxs_;
    int *d_atomic_radii_idxs_;
    int *d_scale_factor_idxs_;
    int *d_lambda_idxs_;

    double *d_du_dl_buffer_;

    unsigned long long *d_born_psi_buffer_;
    double *d_born_radii_buffer_;
    double *d_obc_buffer_;
    unsigned long long *d_born_forces_buffer_;

    Surreal<double> *d_born_radii_buffer_jvp_;
    Surreal<double> *d_obc_buffer_jvp_;
    Surreal<double> *d_obc_ri_buffer_jvp_;
    Surreal<double> *d_born_forces_buffer_jvp_;

    double alpha_;
    double beta_;
    double gamma_;
    double dielectric_offset_;
    double surface_tension_;
    double solute_dielectric_;
    double solvent_dielectric_;
    double probe_radius_;
    double cutoff_radii_;
    double cutoff_force_;

    Neighborlist nblist_;

    const int N_;

public:

    GBSA(
        const std::vector<int> &charge_param_idxs,
        const std::vector<int> &atomic_radii_idxs,
        const std::vector<int> &scale_factor_idxs,
        const std::vector<int> &lambda_idxs, // [N]
        double alpha,
        double beta,
        double gamma,
        double dielectric_offset,
        double surface_tension,
        double solute_dielectric,
        double solvent_dielectric,
        double probe_radius,
        double cutoff_radii,
        double cutoff_force
    );

    // FIX ME with actual destructors later
    ~GBSA();

    virtual void execute_lambda_device(
        const int N,
        const int P,
        const double *d_coords_primals,
        const double *d_coords_tangents,
        const double *d_params_primals,
        const double lambda_primal,
        const double lambda_tangent,
        unsigned long long *d_out_coords_primals,
        double *d_out_lambda_primals,
        double *d_out_coords_tangents,
        double *d_out_params_tangents,
        cudaStream_t stream
    ) override;

};


}