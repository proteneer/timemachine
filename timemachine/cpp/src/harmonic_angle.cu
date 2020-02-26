#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "harmonic_angle.hpp"
#include "kernel_utils.cuh"
#include "k_bonded_deterministic.cuh"

namespace timemachine {

template <typename RealType, int D>
HarmonicAngle<RealType, D>::HarmonicAngle(
    const std::vector<int> &angle_idxs, // [A, 3]
    const std::vector<int> &param_idxs // [A, 2]
) : A_(angle_idxs.size()/3) {

    if(angle_idxs.size() % 3 != 0) {
        throw std::runtime_error("angle_idxs.size() must be exactly 3*k");
    }

    for(int a=0; a < A_; a++) {
        auto i = angle_idxs[a*3+0];
        auto j = angle_idxs[a*3+1];
        auto k = angle_idxs[a*3+2];
        if(i == j || j == k || i == k) {
            throw std::runtime_error("angle triplets must be unique");
        }
    }

    gpuErrchk(cudaMalloc(&d_angle_idxs_, A_*3*sizeof(*d_angle_idxs_)));
    gpuErrchk(cudaMemcpy(d_angle_idxs_, &angle_idxs[0], A_*3*sizeof(*d_angle_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_param_idxs_, A_*3*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], A_*3*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType, int D>
HarmonicAngle<RealType, D>::~HarmonicAngle() {
    gpuErrchk(cudaFree(d_angle_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
};

template <typename RealType, int D>
void HarmonicAngle<RealType, D>::execute_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_coords_tangents,
    const double *d_params,
    unsigned long long *d_out_coords,
    double *d_out_coords_tangents,
    double *d_out_params_tangents,
    cudaStream_t stream
) {

    int tpb = 32;
    int blocks = (A_+tpb-1)/tpb;

    auto start = std::chrono::high_resolution_clock::now();
    if(d_coords_tangents == nullptr) {

        k_harmonic_angle_inference<RealType, D><<<blocks, tpb, 0, stream>>>(
            A_,
            d_coords,
            d_params,
            d_angle_idxs_,
            d_param_idxs_,
            d_out_coords
        );


        gpuErrchk(cudaPeekAtLastError());

        // cudaDeviceSynchronize();
        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "HarmonicAngle Elapsed time: " << elapsed.count() << " s\n";

    } else {


        k_harmonic_angle_jvp<RealType, D><<<blocks, tpb,  0, stream>>>(
            A_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_angle_idxs_,
            d_param_idxs_,
            d_out_coords_tangents,
            d_out_params_tangents
        );


        gpuErrchk(cudaPeekAtLastError());

        // cudaDeviceSynchronize();
        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "HarmonicAngle JVP Elapsed time: " << elapsed.count() << " s\n";


    }


};

template class HarmonicAngle<double, 4>;
template class HarmonicAngle<double, 3>;

template class HarmonicAngle<float, 4>;
template class HarmonicAngle<float, 3>;

} // namespace timemachine