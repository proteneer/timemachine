#include "optimizer.hpp"

namespace timemachine {

template <typename RealType>
class LangevinOptimizer : public Optimizer {

private:

    RealType dt_;
    long long step_;

    // RealType *d_x_t_; // geometries
    // RealType *d_v_t_; // velocities
    // RealType *d_dxdp_t_; // derivatives of geometry wrt parameters
    // RealType *d_dvdp_t_; // derivatives of geometry wrt parameters

    cublasHandle_t cb_handle_;
    curandGenerator_t  cr_rng_;

    RealType *d_rng_buffer_;

    RealType coeff_a_;
    RealType *d_coeff_bs_;
    RealType *d_coeff_cs_;

public:



    LangevinOptimizer(
        RealType dt,
        const RealType coeff_a,
        const std::vector<RealType> &coeff_bs,
        const std::vector<RealType> &coeff_cs
    );

    virtual void step(
        const int num_atoms,
        const int num_params,
        const RealType *dE_dx,
        const RealType *d2E_dx2,
        const RealType *d2E_dxdp,
        RealType *d_x_t, // mutable
        RealType *d_v_t, // mutable
        RealType *d_dx_dp_t, // mutable
        RealType *d_dv_dp_t // mutable
    );

//     virtual void opt_init(
//         const int num_atoms,
//         const RealType *h_x0,
//         const RealType *h_v0
//         const std::vector<int> &dp_idxs;
//     ) override;

//     virtual void apply_fn(
//         const RealType *d_dE_dx, // simple grads
//         const RealType *d_full_d2E_dxdp  // hessian matrix product
//     ) override;

// private:


  // friend class Context;

}

}