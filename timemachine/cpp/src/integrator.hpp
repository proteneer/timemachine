#include <vector>
#include "gradient.hpp"

namespace timemachine {

template<typename RealType>
void step_forward(
    int N,
    int D,
    const RealType ca,
    const RealType *d_coeff_bs,
    const RealType *d_x_old,
    const RealType *d_v_old,
    const unsigned long long *d_dE_dx,
    const RealType dt,
    // const RealType lambda,
    // const int *lambda_flags,
    RealType *d_x_new,
    RealType *d_v_new);


// template<typename RealType>
// void step_backward(
//     int N,
//     int D,
//     const RealType ca,
//     const RealType *d_coeff_bs,
//     const RealType *d_adjoint_x_new,
//     const RealType *d_adjoint_v_new,
//     const RealType *d_adjoint_dE_dx,
//     const RealType *d_adjoint_dE_dp,
//     const RealType dt,
//     const RealType lambda,
//     const int *lambda_flags,
//     RealType *d_adjoint_params,
//     RealType *d_adjoint_x_old,
//     RealType *d_adjoint_v_old);


} // end namespace timemachine