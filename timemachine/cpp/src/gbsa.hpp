#pragma once

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
    double cutoff_;

    // these buffers can be in RealType as well
    // double *d_block_bounds_ctr_;
    // double *d_block_bounds_ext_;

    const int N_;

public:

    GBSA(
        const std::vector<int> &charge_param_idxs, // [N]
        const std::vector<int> &atomic_radii_idxs, // [N]
        const std::vector<int> &scale_factor_idxs, // 
        double alpha,
        double beta,
        double gamma,
        double dielectric_offset,
        // double screening,
        double surface_tension,
        double solute_dielectric,
        double solvent_dielectric,
        double probe_radius,
        double cutoff
    );

    // FIX ME with actual destructors later
    ~GBSA();

    void execute_first_order(
        const int N,
        const int P,
        const std::vector<double> &coords,
        const std::vector<double> &params,
        std::vector<double> &dU_dx,
        std::vector<double> &dU_dp
    );

    void execute_second_order(
        const int N,
        const int P,
        const std::vector<double> &coords,
        const std::vector<double> &coords_tangents,
        const std::vector<double> &params,
        std::vector<double> &HvP,
        std::vector<double> &MvP
    );

    /*
    Execute the force computation, the semantics are:

    1. If d_coords_tangents == null, then out_coords != null, out_coords_tangent == null, out_params_tangents == null
    2. If d_coords_tangents != null, then out_coords == null, out_coords_tangent != null, out_params_tangents != null

    */
    virtual void execute_device(
        const int N,
        const int P,
        const double *d_coords,
        const double *d_coords_tangents,
        const double *d_params,
        unsigned long long *out_coords,
        double *out_coords_tangents,
        double *out_params_tangents
    ) override;


};


}