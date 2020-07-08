#pragma once

#include "neighborlist.hpp"
#include "gradient.hpp"
#include "surreal.cuh"
#include <vector>

namespace timemachine {

template<typename RealType>
class GBSA : public Gradient {

private:

    double *d_charge_params_;
    double *d_gb_params_;

    double *d_du_dcharge_primals_;
    double *d_du_dgb_primals_;
    double *d_du_dcharge_tangents_;
    double *d_du_dgb_tangents_;

    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;

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
        const std::vector<double> &charge_params,
        const std::vector<double> &gb_params,
        const std::vector<int> &lambda_plane_idxs, // N
        const std::vector<int> &lambda_offset_idxs, // N
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

    ~GBSA();

    int num_atoms() const {
        return N_;
    }

    void get_du_dcharge_primals(double *buf);
    void get_du_dcharge_tangents(double *buf);

    void get_du_dgb_primals(double *buf);
    void get_du_dgb_tangents(double *buf);

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