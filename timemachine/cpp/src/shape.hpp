#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class Shape : public Potential {

private:
    const int A_;
    const int B_;
    const int N_;

    int *d_a_idxs_;
    int *d_b_idxs_;

    double *d_alphas_;
    double *d_weights_;

    double *d_f_buffer_;
    double *d_g_buffer_;
    double *d_h_buffer_;

    double *d_df_dx_buffer_;
    double *d_dg_dx_buffer_;
    double *d_dh_dx_buffer_;

    double k_;

public:
    Shape(
        const int N,
        const std::vector<int> &a_idxs,
        const std::vector<int> &b_idxs,
        const std::vector<double> &alphas,
        const std::vector<double> &weights,
        double k);

    ~Shape();

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
        cudaStream_t stream) override;
};

} // namespace timemachine
