#pragma once

#include "optimizer.hpp"

namespace timemachine {


// immutable, stateless class with only const methods
template <typename RealType>
class LangevinOptimizer : public Optimizer<RealType> {

private:

    RealType dt_;

    cublasHandle_t cb_handle_;
    curandGenerator_t  cr_rng_;

    RealType *d_rng_buffer_;
    RealType coeff_a_;
    RealType *d_coeff_bs_;
    RealType *d_coeff_cs_;
    RealType coeff_d_;

    // refactor this to base class later
    void hessian_vector_product(
        const int N,
        const int D,
        const int P,
        const RealType *d_A,
        RealType *d_B,
        RealType *d_C) const;

public:

    virtual ~LangevinOptimizer();

    LangevinOptimizer(
        RealType dt,
        const int num_dims,
        const RealType coeff_a,
        const std::vector<RealType> &coeff_bs,
        const std::vector<RealType> &coeff_cs
    );

    void set_coeff_a(RealType a);

    void set_coeff_b(int num_atoms, const RealType *cb);

    void set_coeff_c(int num_atoms, const RealType *cc);

    void set_coeff_d(RealType d);

    void set_dt(RealType ndt);

    virtual void step(
        const int num_atoms,
        const int num_dims,
        const int num_params,
        const RealType *dE_dx,
        const RealType *d2E_dx2,
        RealType *d2E_dxdp, // this is modified in place
        RealType *d_x_t, // mutable
        RealType *d_v_t, // mutable
        RealType *d_dx_dp_t, // mutable
        RealType *d_dv_dp_t, // mutable
        const RealType *d_input_noise_buffer=nullptr
    ) const override;

};

}