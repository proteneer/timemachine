#pragma once

#include "neighborlist.hpp"
#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class LennardJones : public Gradient {

private:

    double *d_lj_params_; // [N, 2]

    double *d_du_dlj_primals_;
    double *d_du_dlj_tangents_;

    int *d_exclusion_idxs_; // [E,2]
    double *d_lj_scales_; // [E]
    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;
    int *d_lambda_group_idxs_;

    double cutoff_;
    Neighborlist nblist_;

    const int E_;
    const int N_;

public:

    LennardJones(
        const std::vector<double> &lj_params, // [N, 2]
        const std::vector<int> &exclusion_idxs, // [E,2]
        const std::vector<double> &lj_scales, // [E]
        const std::vector<int> &lambda_plane_idxs, // N
        const std::vector<int> &lambda_offset_idxs, // N
        const std::vector<int> &lambda_group_idxs, // N
        double cutoff
    );

    ~LennardJones();

    int num_atoms() const {
        return N_;
    }

    void get_du_dlj_primals(double *buf);
    void get_du_dlj_tangents(double *buf);

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