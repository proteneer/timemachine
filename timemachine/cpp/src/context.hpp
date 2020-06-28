#include <vector>
#include "gradient.hpp"
#include "stepper.hpp"
#include "curand.h"

namespace timemachine {

// we will always integrate in three dimensions
// free energy integrator
class ReversibleContext {

private:

    Stepper *stepper_;

    const std::vector<double> coeff_cas_; // [T]
    const std::vector<double> step_sizes_; // [T]

    // double *d_params_; // [T]
    // double *d_params_grads_;

    curandGenerator_t  cr_rng_;

    double *d_noise_buffer_;

    double *d_coeff_cbs_; // [N]
    double *d_coeff_ccs_; // [N]

    double *d_coords_; // [TxNxD]

    double *d_velocities_; // [NxD]
    unsigned long long *d_forces_; // [NxD], change this to uint128

    double *d_x_t_tangent_; // [NxD]
    // double *d_param_adjoint_accum_; // [P]
    double *d_x_t_adjoint_; // [NxD]
    double *d_v_t_adjoint_; // [NxD]

    double *d_dE_dx_jvp_primals_; // [NxD]
    double *d_dE_dx_jvp_tangents_; // [NxD]

    // double *d_dE_dp_jvp_primals_; // [P]
    // double *d_dE_dp_jvp_tangents_; // [P]

    const int N_;
    // const int P_;
    const int D = 3;

public:

    ReversibleContext(
        Stepper *stepper_,
        const int N,
        const std::vector<double> &x0,
        const std::vector<double> &v0,
        const std::vector<double> &coeff_cas,
        const std::vector<double> &coeff_cbs,
        const std::vector<double> &coeff_ccs,
        const std::vector<double> &dts,
        // const std::vector<double> &params,
        unsigned long long seed
    );

    ~ReversibleContext();

    void get_all_coords(double *out_buffer) const;
    void get_last_coords(double *out_buffer) const;

    // void get_param_adjoint_accum(double *out_buffer) const;
    void set_x_t_adjoint(const double *buffer);
    void get_x_t_adjoint(double *buffer) const;
    void get_v_t_adjoint(double *buffer) const;
    void forward_mode();
    void backward_mode();

    size_t N() const {
        return N_;
    }

    // size_t P() const {
    //     return P_;
    // }

    size_t F() const {
        return T() + 1;
    }

    size_t T() const {
        return step_sizes_.size();
    }

};



}