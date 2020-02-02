#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "periodic_torsion.hpp"
#include "kernel_utils.cuh"
#include "k_bonded_deterministic.cuh"

namespace timemachine {

template <typename RealType, int D>
PeriodicTorsion<RealType, D>::PeriodicTorsion(
    const std::vector<int> &torsion_idxs, // [A, 4]
    const std::vector<int> &param_idxs // [A, 3]
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

    gpuErrchk(cudaMalloc(&d_param_idxs_, T_*3*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], T_*3*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType, int D>
PeriodicTorsion<RealType, D>::~PeriodicTorsion() {
    gpuErrchk(cudaFree(d_torsion_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
};

template <typename RealType, int D>
void PeriodicTorsion<RealType, D>::execute_device(
    const int N,
    const int P,
    const RealType *d_coords,
    const RealType *d_coords_tangents,
    const RealType *d_params,
    unsigned long long *d_out_coords,
    RealType *d_out_coords_tangents,
    RealType *d_out_params_tangents
) {

    int tpb = 32;
    int blocks = (T_+tpb-1)/tpb;

    auto start = std::chrono::high_resolution_clock::now();
    if(d_coords_tangents == nullptr) {

        k_periodic_torsion_inference<RealType, D><<<blocks, tpb>>>(
            T_,
            d_coords,
            d_params,
            d_torsion_idxs_,
            d_param_idxs_,
            d_out_coords
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "PeriodicTorsion Elapsed time: " << elapsed.count() << " s\n";

    } else {


        k_periodic_torsion_jvp<RealType, D><<<blocks, tpb>>>(
            T_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_torsion_idxs_,
            d_param_idxs_,
            d_out_coords_tangents,
            d_out_params_tangents
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "PeriodicTorsion JVP Elapsed time: " << elapsed.count() << " s\n";


    }


};

template class PeriodicTorsion<double, 4>;
template class PeriodicTorsion<double, 3>;

template class PeriodicTorsion<float, 4>;
template class PeriodicTorsion<float, 3>;

} // namespace timemachine