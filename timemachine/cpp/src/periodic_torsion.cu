#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "periodic_torsion.hpp"
#include "gpu_utils.cuh"
#include "k_periodic_torsion.cuh"

namespace timemachine {

template <typename RealType>
PeriodicTorsion<RealType>::PeriodicTorsion(
    const std::vector<int> &torsion_idxs // [A, 4]
) : T_(torsion_idxs.size()/4) {

    if(torsion_idxs.size() % 4 != 0) {
        throw std::runtime_error("torsion_idxs.size() must be exactly 4*k");
    }

    for(int a=0; a < T_; a++) {
        auto i = torsion_idxs[a*4+0];
        auto j = torsion_idxs[a*4+1];
        auto k = torsion_idxs[a*4+2];
        auto l = torsion_idxs[a*4+3];
        if(i == j || i == k || i == l || j == k || j == l || k == l) {
            throw std::runtime_error("torsion quads must be unique");
        }
    }

    gpuErrchk(cudaMalloc(&d_torsion_idxs_, T_*4*sizeof(*d_torsion_idxs_)));
    gpuErrchk(cudaMemcpy(d_torsion_idxs_, &torsion_idxs[0], T_*4*sizeof(*d_torsion_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
PeriodicTorsion<RealType>::~PeriodicTorsion() {
    gpuErrchk(cudaFree(d_torsion_idxs_));
};

template <typename RealType>
void PeriodicTorsion<RealType>::execute_device(
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
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (T_+tpb-1)/tpb;

    const int D = 3;

    if(blocks > 0) {

        k_periodic_torsion<RealType, D><<<blocks, tpb, 0, stream>>>(
            T_,
            d_x,
            d_p,
            d_torsion_idxs_,
            d_du_dx,
            d_du_dp,
            d_u
        );        

        gpuErrchk(cudaPeekAtLastError());

    }

    // (remove me)
    cudaDeviceSynchronize();


};

// template <typename RealType>
// void PeriodicTorsion<RealType>::execute_lambda_jvp_device(
//     const int N,
//     const double *d_coords_primals,
//     const double *d_coords_tangents,
//     const double lambda_primal, // unused
//     const double lambda_tangent, // unused
//     double *d_out_coords_primals,
//     double *d_out_coords_tangents,
//     cudaStream_t stream) {

//     int tpb = 32;
//     int blocks = (T_+tpb-1)/tpb;
//     const int D = 3;
//     k_periodic_torsion_jvp<RealType, D><<<blocks, tpb, 0, stream>>>(
//         T_,
//         d_coords_primals,
//         d_coords_tangents,
//         d_params_,
//         d_torsion_idxs_,
//         d_out_coords_primals,
//         d_out_coords_tangents,
//         d_du_dp_primals_,
//         d_du_dp_tangents_
//     );

//     cudaDeviceSynchronize();
//     gpuErrchk(cudaPeekAtLastError());

// };

template class PeriodicTorsion<double>;
template class PeriodicTorsion<float>;

} // namespace timemachine