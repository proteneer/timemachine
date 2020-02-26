#include <stdexcept>
#include <iostream> 
#include <chrono>
#include "fixed_point.hpp"
#include "gbsa.hpp"
#include "gbsa_jvp.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "k_gbsa.cuh"
#include "k_gbsa_jvp.cuh"

namespace timemachine {

template <typename RealType, int D>
GBSA<RealType, D>::GBSA(
    const std::vector<int> &charge_param_idxs, // [N]
    const std::vector<int> &atomic_radii_idxs, // [N]
    const std::vector<int> &scale_factor_idxs, // [E,2]
    double alpha,
    double beta,
    double gamma,
    double dielectric_offset,
    double surface_tension,
    double solute_dielectric,
    double solvent_dielectric,
    double probe_radius,
    double cutoff) :

    N_(charge_param_idxs.size()),
    alpha_(alpha),
    beta_(beta),
    gamma_(gamma),
    dielectric_offset_(dielectric_offset),
    surface_tension_(surface_tension),
    solute_dielectric_(solute_dielectric),
    solvent_dielectric_(solvent_dielectric),
    probe_radius_(probe_radius),
    cutoff_(cutoff) {

    gpuErrchk(cudaMalloc(&d_charge_param_idxs_, N_*sizeof(*d_charge_param_idxs_)));
    gpuErrchk(cudaMalloc(&d_scale_factor_idxs_, N_*sizeof(*d_scale_factor_idxs_)));
    gpuErrchk(cudaMalloc(&d_atomic_radii_idxs_, N_*sizeof(*d_atomic_radii_idxs_)));
    gpuErrchk(cudaMemcpy(d_charge_param_idxs_, &charge_param_idxs[0], N_*sizeof(*d_charge_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_factor_idxs_, &scale_factor_idxs[0], N_*sizeof(*d_scale_factor_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_atomic_radii_idxs_, &atomic_radii_idxs[0], N_*sizeof(*d_atomic_radii_idxs_), cudaMemcpyHostToDevice));

    // we probaly don't need *all* these buffers if we do just one pass, but they take up only
    // O(N) ram so we don't really care and just pre-allocate everything to keep things simple.
    // it also ensures that we can RAII properly.

    const int N = charge_param_idxs.size();

    gpuErrchk(cudaMalloc(&d_born_psi_buffer_, N*sizeof(*d_born_psi_buffer_)));
    gpuErrchk(cudaMalloc(&d_born_radii_buffer_, N*sizeof(*d_born_radii_buffer_)));
    gpuErrchk(cudaMalloc(&d_obc_buffer_, N*sizeof(*d_obc_buffer_)));
    gpuErrchk(cudaMalloc(&d_born_forces_buffer_, N*sizeof(*d_born_forces_buffer_)));

    gpuErrchk(cudaMalloc(&d_born_radii_buffer_jvp_, N*sizeof(*d_born_radii_buffer_jvp_)));
    gpuErrchk(cudaMalloc(&d_obc_buffer_jvp_, N*sizeof(*d_obc_buffer_jvp_)));
    gpuErrchk(cudaMalloc(&d_obc_ri_buffer_jvp_, N*sizeof(*d_obc_ri_buffer_jvp_)));
    gpuErrchk(cudaMalloc(&d_born_forces_buffer_jvp_, N*sizeof(*d_born_forces_buffer_jvp_)));


}

template <typename RealType, int D>
GBSA<RealType, D>::~GBSA() {

  gpuErrchk(cudaFree(d_charge_param_idxs_));
  gpuErrchk(cudaFree(d_atomic_radii_idxs_));
  gpuErrchk(cudaFree(d_scale_factor_idxs_));

  gpuErrchk(cudaFree(d_born_psi_buffer_));
  gpuErrchk(cudaFree(d_born_radii_buffer_));
  gpuErrchk(cudaFree(d_obc_buffer_));
  gpuErrchk(cudaFree(d_born_forces_buffer_));

  gpuErrchk(cudaFree(d_born_radii_buffer_jvp_));
  gpuErrchk(cudaFree(d_obc_buffer_jvp_));
  gpuErrchk(cudaFree(d_obc_ri_buffer_jvp_));
  gpuErrchk(cudaFree(d_born_forces_buffer_jvp_));


};


template <typename RealType, int D>
void GBSA<RealType, D>::execute_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_coords_tangents,
    const double *d_params,
    unsigned long long *d_out_coords,
    double *d_out_coords_tangents,
    double *d_out_params_tangents
) {

    int tpb = 32;
    int B = (N_+tpb-1)/tpb;

    dim3 dimGrid(B, B, 1); // x, y, z dims

    double prefactor;
    if (solute_dielectric_ != 0.0 && solvent_dielectric_ != 0.0) {
        // prefactor = -screening_*((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
        prefactor = -((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
    } else {
        prefactor = 0.0;
    }

    // inference mode
    if(d_coords_tangents == nullptr) {
        gpuErrchk(cudaMemset(d_born_psi_buffer_, 0, N*sizeof(*d_born_psi_buffer_)));
        gpuErrchk(cudaMemset(d_born_radii_buffer_, 0, N*sizeof(*d_born_radii_buffer_)));
        gpuErrchk(cudaMemset(d_obc_buffer_, 0, N*sizeof(*d_obc_buffer_)));
        gpuErrchk(cudaMemset(d_born_forces_buffer_, 0, N*sizeof(*d_born_forces_buffer_)));

        k_compute_born_radii_gpu<RealType, D><<<dimGrid, tpb>>>(
          N_,
          d_coords,
          d_params,
          d_atomic_radii_idxs_,
          d_scale_factor_idxs_,
          dielectric_offset_,
          cutoff_,
          d_born_psi_buffer_
        );


        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_reduce_born_radii<<<B, tpb>>>(
          N_,
          d_params,
          d_atomic_radii_idxs_,
          dielectric_offset_,
          alpha_,
          beta_,
          gamma_,
          d_born_psi_buffer_,
          d_born_radii_buffer_,
          d_obc_buffer_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());


        k_compute_born_first_loop_gpu<RealType, D><<<dimGrid, tpb>>>(
          N_,
          d_coords,
          d_params,
          d_charge_param_idxs_,
          d_born_radii_buffer_,
          prefactor,
          cutoff_,
          d_born_forces_buffer_, // output
          d_out_coords // ouput
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_reduce_born_forces<<<B, tpb>>>(
          N_,
          d_params,
          d_atomic_radii_idxs_,
          d_born_radii_buffer_,
          d_obc_buffer_,
          surface_tension_,
          probe_radius_,
          d_born_forces_buffer_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());


        auto start = std::chrono::high_resolution_clock::now();
        k_compute_born_energy_and_forces<RealType, D><<<dimGrid, tpb>>>(
          N_,
          d_coords,
          d_params,
          d_atomic_radii_idxs_,
          d_scale_factor_idxs_,
          d_born_radii_buffer_,
          d_obc_buffer_,
          dielectric_offset_,
          cutoff_,
          d_born_forces_buffer_,
          d_out_coords
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "Nonbonded Elapsed time: " << elapsed.count() << " s\n";

    } else {
        gpuErrchk(cudaMemset(d_born_radii_buffer_jvp_, 0, N*sizeof(*d_born_radii_buffer_jvp_)));
        gpuErrchk(cudaMemset(d_obc_buffer_jvp_, 0, N*sizeof(*d_obc_buffer_jvp_)));
        gpuErrchk(cudaMemset(d_obc_ri_buffer_jvp_, 0, N*sizeof(*d_obc_ri_buffer_jvp_)));
        gpuErrchk(cudaMemset(d_born_forces_buffer_jvp_, 0, N*sizeof(*d_born_forces_buffer_jvp_)));

        k_compute_born_radii_gpu_jvp<RealType, D><<<dimGrid, tpb>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_atomic_radii_idxs_,
            d_scale_factor_idxs_,
            dielectric_offset_,
            cutoff_,
            d_born_radii_buffer_jvp_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_reduce_born_radii_jvp<<<B, tpb>>>(
          N_,
          d_params,
          d_atomic_radii_idxs_,
          dielectric_offset_,
          alpha_,
          beta_,
          gamma_,
          d_born_radii_buffer_jvp_,
          d_obc_buffer_jvp_,
          d_obc_ri_buffer_jvp_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_compute_born_first_loop_gpu_jvp<RealType, D><<<dimGrid, tpb>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_charge_param_idxs_,
            d_born_radii_buffer_jvp_,
            prefactor,
            cutoff_,
            d_born_forces_buffer_jvp_, // output
            d_out_coords_tangents, // ouput
            d_out_params_tangents // ouput
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_reduce_born_forces_jvp<<<B, tpb>>>(
            N_,
            d_params,
            d_atomic_radii_idxs_,
            d_born_radii_buffer_jvp_,
            d_obc_buffer_jvp_,
            d_obc_ri_buffer_jvp_,
            surface_tension_,
            probe_radius_,
            d_born_forces_buffer_jvp_,
            d_out_params_tangents
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());


        auto start = std::chrono::high_resolution_clock::now();
        k_compute_born_energy_and_forces_jvp<RealType, D><<<dimGrid, tpb>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_atomic_radii_idxs_,
            d_scale_factor_idxs_,
            d_born_radii_buffer_jvp_,
            d_obc_buffer_jvp_,
            d_obc_ri_buffer_jvp_,
            dielectric_offset_,
            cutoff_,
            d_born_forces_buffer_jvp_,
            d_out_coords_tangents,
            d_out_params_tangents
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "Nonbonded JVP Elapsed time: " << elapsed.count() << " s\n";


    }

}

template class GBSA<double, 4>;
template class GBSA<double, 3>;

template class GBSA<float, 4>;
template class GBSA<float, 3>;


}

