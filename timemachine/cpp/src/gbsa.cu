#include <stdexcept>
#include <iostream> 
#include <chrono>
#include "fixed_point.hpp"
#include "gbsa.hpp"
#include "gbsa_jvp.cuh"
#include "gpu_utils.cuh"
#include "math_utils.cuh"
#include "k_gbsa.cuh"
#include "k_gbsa_jvp.cuh"

namespace timemachine {

template <typename RealType>
GBSA<RealType>::GBSA(
    const std::vector<int> &charge_param_idxs, // [N]
    const std::vector<int> &atomic_radii_idxs, // [N]
    const std::vector<int> &scale_factor_idxs, // [E,2]
    const std::vector<int> &lambda_plane_idxs, // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double alpha,
    double beta,
    double gamma,
    double dielectric_offset,
    double surface_tension,
    double solute_dielectric,
    double solvent_dielectric,
    double probe_radius,
    double cutoff_radii,
    double cutoff_force
) : N_(charge_param_idxs.size()),
    alpha_(alpha),
    beta_(beta),
    gamma_(gamma),
    dielectric_offset_(dielectric_offset),
    surface_tension_(surface_tension),
    solute_dielectric_(solute_dielectric),
    solvent_dielectric_(solvent_dielectric),
    probe_radius_(probe_radius),
    cutoff_radii_(cutoff_radii),
    cutoff_force_(cutoff_force),
    nblist_(charge_param_idxs.size(), 3) {

    if(cutoff_radii != cutoff_force) {
      throw std::runtime_error("GB currently requires that cutoff_radii be equal to cutoff_force!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_*sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_*sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

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

template <typename RealType>
GBSA<RealType>::~GBSA() {

  gpuErrchk(cudaFree(d_charge_param_idxs_));
  gpuErrchk(cudaFree(d_atomic_radii_idxs_));
  gpuErrchk(cudaFree(d_scale_factor_idxs_));
  gpuErrchk(cudaFree(d_lambda_plane_idxs_));
  gpuErrchk(cudaFree(d_lambda_offset_idxs_));

  gpuErrchk(cudaFree(d_born_psi_buffer_));
  gpuErrchk(cudaFree(d_born_radii_buffer_));
  gpuErrchk(cudaFree(d_obc_buffer_));
  gpuErrchk(cudaFree(d_born_forces_buffer_));

  gpuErrchk(cudaFree(d_born_radii_buffer_jvp_));
  gpuErrchk(cudaFree(d_obc_buffer_jvp_));
  gpuErrchk(cudaFree(d_obc_ri_buffer_jvp_));
  gpuErrchk(cudaFree(d_born_forces_buffer_jvp_));

};

template <typename RealType>
void GBSA<RealType>::execute_lambda_inference_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_params,
    const double lambda,
    unsigned long long *d_out_coords,
    double *d_out_lambda,
    double *d_out_energy,
    cudaStream_t stream) {

    // std::cout << "exec GB" << std::endl;

    int tpb = 32;
    int B = (N_+tpb-1)/tpb;
    const int D = 3;

    dim3 dimGrid(B, B, 1); // x, y, z dims

    double prefactor;
    if (solute_dielectric_ != 0.0 && solvent_dielectric_ != 0.0) {
        // prefactor = -screening_*((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
        prefactor = -((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
    } else {
        prefactor = 0.0;
    }
    // std::cout << "cutoff 12: " << cutoff_radii_ << " " << cutoff_force_ << std::endl;

    // cudaDeviceSynchronize();
    nblist_.compute_block_bounds(N_, D, d_coords, stream);

    auto start = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemsetAsync(d_born_psi_buffer_, 0, N*sizeof(*d_born_psi_buffer_), stream));
    gpuErrchk(cudaMemsetAsync(d_born_radii_buffer_, 0, N*sizeof(*d_born_radii_buffer_), stream));
    gpuErrchk(cudaMemsetAsync(d_obc_buffer_, 0, N*sizeof(*d_obc_buffer_), stream));
    gpuErrchk(cudaMemsetAsync(d_born_forces_buffer_, 0, N*sizeof(*d_born_forces_buffer_), stream));

    k_compute_born_radii<RealType><<<dimGrid, tpb, 0, stream>>>(
      N_,
      d_coords,
      d_params,
      lambda,
      d_lambda_plane_idxs_,
      d_lambda_offset_idxs_,
      d_atomic_radii_idxs_,
      d_scale_factor_idxs_,
      dielectric_offset_,
      cutoff_radii_,
      nblist_.get_block_bounds_ctr(),
      nblist_.get_block_bounds_ext(),
      d_born_psi_buffer_
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_reduce_born_radii<<<B, tpb, 0, stream>>>(
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

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_compute_born_first_loop_gpu<RealType><<<dimGrid, tpb, 0, stream>>>(
      N_,
      d_coords,
      d_params,
      lambda,
      d_lambda_plane_idxs_,
      d_lambda_offset_idxs_,
      d_charge_param_idxs_,
      d_born_radii_buffer_,
      prefactor,
      cutoff_force_,
      nblist_.get_block_bounds_ctr(),
      nblist_.get_block_bounds_ext(),
      d_born_forces_buffer_, // output
      d_out_coords,
      d_out_lambda, // output
      d_out_energy
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_reduce_born_forces<<<B, tpb, 0, stream>>>(
      N_,
      d_params,
      d_atomic_radii_idxs_,
      d_born_radii_buffer_,
      d_obc_buffer_,
      surface_tension_,
      probe_radius_,
      d_born_forces_buffer_,
      d_out_energy
    );

    // // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_compute_born_energy_and_forces<RealType><<<dimGrid, tpb, 0, stream>>>(
      N_,
      d_coords,
      d_params,
      lambda,
      d_lambda_plane_idxs_,
      d_lambda_offset_idxs_,
      d_atomic_radii_idxs_,
      d_scale_factor_idxs_,
      d_born_radii_buffer_,
      d_obc_buffer_,
      dielectric_offset_,
      cutoff_force_,
      nblist_.get_block_bounds_ctr(),
      nblist_.get_block_bounds_ext(),
      d_born_forces_buffer_,
      d_out_coords,
      d_out_lambda
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

}

template <typename RealType>
void GBSA<RealType>::execute_lambda_jvp_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal,
    const double lambda_tangent,
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    double *d_out_params_primals,
    double *d_out_params_tangents,
    cudaStream_t stream) {

    // std::cout << "exec GB" << std::endl;

    int tpb = 32;
    int B = (N_+tpb-1)/tpb;
    const int D = 3;

    dim3 dimGrid(B, B, 1); // x, y, z dims

    double prefactor;
    if (solute_dielectric_ != 0.0 && solvent_dielectric_ != 0.0) {
        prefactor = -((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
    } else {
        prefactor = 0.0;
    }

    nblist_.compute_block_bounds(N_, D, d_coords_primals, stream);

    auto start = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemsetAsync(d_born_radii_buffer_jvp_, 0, N*sizeof(*d_born_radii_buffer_jvp_), stream));
    gpuErrchk(cudaMemsetAsync(d_obc_buffer_jvp_, 0, N*sizeof(*d_obc_buffer_jvp_), stream));
    gpuErrchk(cudaMemsetAsync(d_obc_ri_buffer_jvp_, 0, N*sizeof(*d_obc_ri_buffer_jvp_), stream));
    gpuErrchk(cudaMemsetAsync(d_born_forces_buffer_jvp_, 0, N*sizeof(*d_born_forces_buffer_jvp_), stream));

    k_compute_born_radii_gpu_jvp<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        lambda_primal,
        lambda_tangent,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_atomic_radii_idxs_,
        d_scale_factor_idxs_,
        dielectric_offset_,
        cutoff_radii_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_born_radii_buffer_jvp_
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_reduce_born_radii_jvp<<<B, tpb, 0, stream>>>(
      N_,
      d_params_primals,
      d_atomic_radii_idxs_,
      dielectric_offset_,
      alpha_,
      beta_,
      gamma_,
      d_born_radii_buffer_jvp_,
      d_obc_buffer_jvp_,
      d_obc_ri_buffer_jvp_
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_compute_born_first_loop_gpu_jvp<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        lambda_primal,
        lambda_tangent,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_charge_param_idxs_,
        d_born_radii_buffer_jvp_,
        prefactor,
        cutoff_force_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_born_forces_buffer_jvp_, // output
        d_out_coords_primals, // output
        d_out_coords_tangents, // output
        d_out_params_primals, // output
        d_out_params_tangents // output
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    k_reduce_born_forces_jvp<<<B, tpb, 0, stream>>>(
        N_,
        d_params_primals,
        d_atomic_radii_idxs_,
        d_born_radii_buffer_jvp_,
        d_obc_buffer_jvp_,
        d_obc_ri_buffer_jvp_,
        surface_tension_,
        probe_radius_,
        d_born_forces_buffer_jvp_,
        d_out_params_primals,
        d_out_params_tangents
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    // auto start = std::chrono::high_resolution_clock::now();
    k_compute_born_energy_and_forces_jvp<RealType, D><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        lambda_primal,
        lambda_tangent,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_atomic_radii_idxs_,
        d_scale_factor_idxs_,
        d_born_radii_buffer_jvp_,
        d_obc_buffer_jvp_,
        d_obc_ri_buffer_jvp_,
        dielectric_offset_,
        cutoff_force_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_born_forces_buffer_jvp_,
        d_out_coords_primals, // output
        d_out_coords_tangents, // output
        d_out_params_primals, // output
        d_out_params_tangents // output
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

}

template class GBSA<double>;
template class GBSA<float>;


}

