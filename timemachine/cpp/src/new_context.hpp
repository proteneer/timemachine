#include <vector>
#include "gradient.hpp"
#include "stepper.hpp"

namespace timemachine {

// we will always integrate in three dimensions
// free energy integrator
template <typename RealType, int D>
class ReversibleContext {

private:

    Stepper<RealType> *stepper_;

    const std::vector<RealType> coeff_cas_; // [T]
    const std::vector<RealType> step_sizes_; // [T]

    RealType *d_params_; // [T]
    RealType *d_params_grads_;

    RealType *d_coeff_cbs_; // [N]

    RealType *d_coords_; // [TxNxD]
    RealType *d_du_dls_; // [T]

    RealType *d_velocities_; // [NxD]
    unsigned long long *d_forces_; // [NxD], change this to uint128

    RealType *d_x_t_tangent_; // [NxD]
    RealType *d_param_adjoint_accum_; // [P]
    RealType *d_x_t_adjoint_; // [NxD]
    RealType *d_v_t_adjoint_; // [NxD]

    RealType *d_dE_dx_jvp_; // [NxD]
    RealType *d_dE_dp_jvp_; // [P]

    const int N_;
    const int P_;

public:

    ReversibleContext(
        Stepper<RealType> *stepper_,
        const int N,
        const std::vector<RealType> &x0,
        const std::vector<RealType> &v0,
        const std::vector<RealType> &coeff_cas,
        const std::vector<RealType> &coeff_cbs,
        const std::vector<RealType> &dts,
        const std::vector<RealType> &params
    );

    ~ReversibleContext();

    void get_all_coords(RealType *out_buffer) const;

    void get_param_adjoint_accum(RealType *out_buffer) const;
    void set_x_t_adjoint(const RealType *buffer);
    void get_x_t_adjoint(RealType *buffer) const;
    void get_v_t_adjoint(RealType *buffer) const;
    void forward_mode();
    void backward_mode();

    size_t N() const {
        return N_;
    }

    size_t P() const {
        return P_;
    }

    size_t F() const {
        return T() + 1;
    }

    size_t T() const {
        return step_sizes_.size();
    }

};



}