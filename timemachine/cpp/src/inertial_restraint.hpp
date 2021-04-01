#pragma once

#include "potential.hpp"
#include <vector>

namespace jankmachine {

template<typename RealType>
class InertialRestraint : public Potential {

private:

    int N_;
    int N_A_;
    int N_B_;
    int N_C_;

    const std::vector<int> h_a_idxs_;
    const std::vector<int> h_b_idxs_;
    const std::vector<double> h_masses_;

    std::vector<double> h_conf_adjoint_;

    std::vector<int> h_c_idxs_;
    int *d_c_idxs_;

    std::vector<double> h_x_buffer_; // Nx3

    double *h_x_memcpy_buf_pinned_;
    double *d_x_memcpy_buf_; // [(N_A+N_B)x3]

    double k_; // force constant

public:

    InertialRestraint(
        const std::vector<int> &group_a_idxs,
        const std::vector<int> &group_b_idxs,
        const std::vector<double> &masses,
        const double k
    );

    ~InertialRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) override;

};


}