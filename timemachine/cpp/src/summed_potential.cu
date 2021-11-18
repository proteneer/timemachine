#include "summed_potential.hpp"

namespace timemachine {

SummedPotential::SummedPotential(Potential &u_a, Potential &u_b, const int P_a) : u_a(u_a), u_b(u_b), P_a(P_a){};

void SummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,   // N * 3
    const double *d_p,   // P_a + P_b
    const double *d_box, // 3 * 3
    const double lambda,
    unsigned long long *d_du_dx, // 2 * (N * 3)
    double *d_du_dp,             // P_a + P_b
    unsigned long long *d_du_dl, // 2 * N
    unsigned long long *d_u,     // 2 * N
    cudaStream_t stream) {

    u_a.execute_device(
        N,
        P_a, // number of parameters for first potential
        d_x,
        d_p,
        d_box,
        lambda,
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u,
        stream);

    u_b.execute_device(
        N,
        P - P_a, // number of parameters for second potential
        d_x,
        d_p + P_a,
        d_box,
        lambda,
        d_du_dx + N * 3,
        d_du_dp + P_a,
        d_du_dl + N,
        d_u + N,
        stream);
};

} // namespace timemachine
