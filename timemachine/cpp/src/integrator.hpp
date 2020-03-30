#include <vector>
#include "gradient.hpp"

namespace timemachine {

template<typename RealType>
void step_forward(
    int N,
    int D,
    const RealType ca,
    const RealType *d_coeff_bs,
    const RealType *d_coeff_cs,
    const RealType *d_noise_buf,
    const RealType *d_x_old,
    const RealType *d_v_old,
    const unsigned long long *d_dE_dx,
    const RealType dt,
    RealType *d_x_new,
    RealType *d_v_new);

} // end namespace timemachine