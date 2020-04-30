#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "harmonic_angle.hpp"
#include "gpu_utils.cuh"
#include "k_bonded_deterministic.cuh"

namespace timemachine {

template <typename RealType>
HarmonicAngle<RealType>::HarmonicAngle(
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

template <typename RealType>
HarmonicAngle<RealType>::~HarmonicAngle() {
    gpuErrchk(cudaFree(d_angle_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
};

template <typename RealType>
void HarmonicAngle<RealType>::execute_lambda_inference_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_params_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals, // du/dx
    double *d_out_lambda_primal, // du/dl
    double *d_out_energy_primal, // U
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (A_+tpb-1)/tpb;

    auto start = std::chrono::high_resolution_clock::now();
    // if(d_coords_tangents == nullptr) {

    k_harmonic_angle_inference<RealType, 3><<<blocks, tpb, 0, stream>>>(
        A_,
        d_coords_primals,
        d_params_primals,
        d_angle_idxs_,
        d_param_idxs_,
        d_out_coords_primals,
        d_out_energy_primal
    );

    gpuErrchk(cudaPeekAtLastError());

}



template <typename RealType>
void HarmonicAngle<RealType>::execute_lambda_jvp_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal, // unused
    const double lambda_tangent, // unused
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    double *d_out_params_primals,
    double *d_out_params_tangents,
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (A_+tpb-1)/tpb;

    const int D = 3;

    k_harmonic_angle_jvp<RealType, 3><<<blocks, tpb,  0, stream>>>(
        A_,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        d_angle_idxs_,
        d_param_idxs_,
        d_out_coords_primals,
        d_out_coords_tangents,
        d_out_params_primals,
        d_out_params_tangents
    );

    gpuErrchk(cudaPeekAtLastError());

    //     gpuErrchk(cudaPeekAtLastError());

    //     // cudaDeviceSynchronize();
    //     // auto finish = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double> elapsed = finish - start;
    //     // std::cout << "HarmonicAngle JVP Elapsed time: " << elapsed.count() << " s\n";


    // }


};

template class HarmonicAngle<double>;
template class HarmonicAngle<float>;

} // namespace timemachine