#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cub/cub.cuh>
#include "jitify.hpp"

#include "nonbonded.hpp"
#include "hilbert.h"
#include "gpu_utils.cuh"

#include "k_nonbonded.cuh"

#include <string>
#include <fstream>
#include <streambuf>

#define NONBONDED_KERNEL_BLOCKS 8192
#define HILBERT_SORT_INTERVAL 100

namespace timemachine {

template <typename T>
void __global__ k_arange(int N, T *arr) {
    const int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }
    arr[atom_idx] = atom_idx;
}

template <typename RealType, bool Interpolated>
Nonbonded<RealType, Interpolated>::Nonbonded(
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &scales, // [E, 2]
    const std::vector<int> &lambda_plane_idxs, // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    const double beta,
    const double cutoff,
    const std::string &kernel_src
    // const std::string &transform_lambda_charge,
    // const std::string &transform_lambda_sigma,
    // const std::string &transform_lambda_epsilon,
    // const std::string &transform_lambda_w
) :  N_(lambda_offset_idxs.size()),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_offset_idxs.size()),
    beta_(beta),
    cur_step_(0),
    last_kernel_(0),
    d_sort_storage_(nullptr),
    d_sort_storage_bytes_(0),
    d_partition_storage_(nullptr),
    d_partition_storage_bytes_(0),
    nblist_padding_(0.1),
    disable_hilbert_(false),
    kernel_ptrs_({
        // enumerate over every possible kernel combination
        // U: Compute U
        // X: Compute DU_DL
        // L: Compute DU_DX
        // P: Compute DU_DP
        //                             U  X  L  P
        &k_nonbonded_unified<RealType, 0, 0, 0, 0>,
        &k_nonbonded_unified<RealType, 0, 0, 0, 1>,
        &k_nonbonded_unified<RealType, 0, 0, 1, 0>,
        &k_nonbonded_unified<RealType, 0, 0, 1, 1>,
        &k_nonbonded_unified<RealType, 0, 1, 0, 0>,
        &k_nonbonded_unified<RealType, 0, 1, 0, 1>,
        &k_nonbonded_unified<RealType, 0, 1, 1, 0>,
        &k_nonbonded_unified<RealType, 0, 1, 1, 1>,
        &k_nonbonded_unified<RealType, 1, 0, 0, 0>,
        &k_nonbonded_unified<RealType, 1, 0, 0, 1>,
        &k_nonbonded_unified<RealType, 1, 0, 1, 0>,
        &k_nonbonded_unified<RealType, 1, 0, 1, 1>,
        &k_nonbonded_unified<RealType, 1, 1, 0, 0>,
        &k_nonbonded_unified<RealType, 1, 1, 0, 1>,
        &k_nonbonded_unified<RealType, 1, 1, 1, 0>,
        &k_nonbonded_unified<RealType, 1, 1, 1, 1>
    }),
    compute_w_coords_instance_(kernel_cache_.program(kernel_src.c_str()).kernel("k_compute_w_coords").instantiate()),
    compute_permute_interpolated_(kernel_cache_.program(kernel_src.c_str()).kernel("k_permute_interpolated").instantiate()),
    compute_add_ull_to_real_interpolated_(kernel_cache_.program(kernel_src.c_str()).kernel("k_add_ull_to_real_interpolated").instantiate()) {

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(lambda_offset_idxs.size() != lambda_plane_idxs.size()) {
        throw std::runtime_error("lambda offset idxs and plane idxs need to be equivalent");
    }

    if(scales.size()/2 != E_) {
        throw std::runtime_error("bad scales size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_*sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_*sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_sorted_, N_*sizeof(*d_lambda_offset_idxs_sorted_)));

    gpuErrchk(cudaMalloc(&d_perm_, N_*sizeof(*d_perm_)));

    gpuErrchk(cudaMalloc(&d_sorted_x_, N_*3*sizeof(*d_sorted_x_)));

    gpuErrchk(cudaMalloc(&d_w_, N_*sizeof(*d_w_)));
    gpuErrchk(cudaMalloc(&d_dw_dl_, N_*sizeof(*d_dw_dl_)));
    gpuErrchk(cudaMalloc(&d_sorted_w_, N_*sizeof(*d_sorted_w_)));
    gpuErrchk(cudaMalloc(&d_sorted_dw_dl_, N_*sizeof(*d_sorted_dw_dl_)));

    gpuErrchk(cudaMalloc(&d_unsorted_p_, N_*3*sizeof(*d_unsorted_p_))); // interpolated
    gpuErrchk(cudaMalloc(&d_sorted_p_, N_*3*sizeof(*d_sorted_p_))); // interpolated
    gpuErrchk(cudaMalloc(&d_unsorted_dp_dl_, N_*3*sizeof(*d_unsorted_dp_dl_))); // interpolated
    gpuErrchk(cudaMalloc(&d_sorted_dp_dl_, N_*3*sizeof(*d_sorted_dp_dl_))); // interpolated
    gpuErrchk(cudaMalloc(&d_sorted_du_dx_, N_*3*sizeof(*d_sorted_du_dx_)));
    gpuErrchk(cudaMalloc(&d_sorted_du_dp_, N_*3*sizeof(*d_sorted_du_dp_)));
    gpuErrchk(cudaMalloc(&d_du_dp_buffer_, N_*3*sizeof(*d_du_dp_buffer_)));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_scales_, E_*2*sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], E_*2*sizeof(*d_scales_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_nblist_x_, N_*3*sizeof(*d_nblist_x_)));
    gpuErrchk(cudaMemset(d_nblist_x_, 0, N_*3*sizeof(*d_nblist_x_))); // set non-sensical positions
    gpuErrchk(cudaMalloc(&d_nblist_box_, 3*3*sizeof(*d_nblist_box_)));
    gpuErrchk(cudaMemset(d_nblist_box_, 0, 3*3*sizeof(*d_nblist_box_)));
    gpuErrchk(cudaMalloc(&d_rebuild_nblist_, 1*sizeof(*d_rebuild_nblist_)));

    gpuErrchk(cudaMalloc(&d_sort_keys_in_, N_*sizeof(d_sort_keys_in_)));
    gpuErrchk(cudaMalloc(&d_sort_keys_out_, N_*sizeof(d_sort_keys_out_)));
    gpuErrchk(cudaMalloc(&d_sort_vals_in_, N_*sizeof(d_sort_vals_in_)));

    const int tpb = 32;
    unsigned int MAX_TILES = nblist_.B()*nblist_.B();
    const int X = (MAX_TILES+tpb-1)/tpb;

    gpuErrchk(cudaMalloc(&d_x_last_, N_*3*sizeof(*d_x_last_)));
    gpuErrchk(cudaMalloc(&d_p_last_, N_*3*sizeof(*d_p_last_)));

    gpuErrchk(cudaMalloc(&d_compaction_mask_, MAX_TILES*sizeof(*d_compaction_mask_)));
    gpuErrchk(cudaMalloc(&d_tile_mask_, MAX_TILES*sizeof(*d_tile_mask_)));
    gpuErrchk(cudaMalloc(&d_tile_idxs_, MAX_TILES*sizeof(*d_tile_idxs_)));
    gpuErrchk(cudaMalloc(&d_nblist_tiles_, MAX_TILES*sizeof(*d_nblist_tiles_)));
    gpuErrchk(cudaMalloc(&d_default_tile_mask_, MAX_TILES*sizeof(*d_default_tile_mask_)));
    gpuErrchk(cudaMemset(d_default_tile_mask_, 0, MAX_TILES*sizeof(*d_default_tile_mask_)));

    gpuErrchk(cudaMalloc(&d_compacted_ixn_count_, 1*sizeof(*d_compacted_ixn_count_)));
    gpuErrchk(cudaMemset(d_compacted_ixn_count_, 0, 1*sizeof(*d_compacted_ixn_count_)));

    gpuErrchk(cudaMalloc(&d_alchemical_u_, N_*sizeof(*d_alchemical_u_)));
    gpuErrchk(cudaMalloc(&d_vanilla_u_, N_*sizeof(*d_vanilla_u_)));

    gpuErrchk(cudaMalloc(&d_alchemical_du_dx_, 3*N_*sizeof(*d_alchemical_du_dx_)));
    gpuErrchk(cudaMalloc(&d_vanilla_du_dx_, 3*N_*sizeof(*d_vanilla_du_dx_)));

    gpuErrchk(cudaMalloc(&d_alchemical_du_dp_, 3*N_*sizeof(*d_alchemical_du_dp_)));
    gpuErrchk(cudaMalloc(&d_vanilla_du_dp_, 3*N_*sizeof(*d_vanilla_du_dp_)));

    gpuErrchk(cudaMalloc(&d_alchemical_du_dl_, N_*sizeof(*d_alchemical_du_dl_)));
    gpuErrchk(cudaMalloc(&d_vanilla_du_dl_, N_*sizeof(*d_vanilla_du_dl_)));

    gpuErrchk(cudaMalloc(&d_run_vanilla_tiles_, 1*sizeof(*d_run_vanilla_tiles_)));
    gpuErrchk(cudaMemset(d_run_vanilla_tiles_, 0, 1*sizeof(*d_run_vanilla_tiles_)));


    k_arange<<<X, tpb>>>(MAX_TILES, d_tile_idxs_);
    gpuErrchk(cudaPeekAtLastError());

    // initialize hilbert curve
    std::vector<unsigned int> bin_to_idx(256*256*256);
    for(int i=0; i < 256; i++) {
        for(int j=0; j < 256; j++) {
            for(int k=0; k < 256; k++) {

                bitmask_t hilbert_coords[3];
                hilbert_coords[0] = i;
                hilbert_coords[1] = j;
                hilbert_coords[2] = k;

                unsigned int bin = static_cast<unsigned int>(hilbert_c2i(3, 8, hilbert_coords));
                bin_to_idx[i*256*256 + j*256 + k] = bin;

            }
        }
    }

    gpuErrchk(cudaMalloc(&d_bin_to_idx_, 256*256*256*sizeof(*d_bin_to_idx_)));
    gpuErrchk(cudaMemcpy(d_bin_to_idx_, &bin_to_idx[0], 256*256*256*sizeof(*d_bin_to_idx_), cudaMemcpyHostToDevice));

    // estimate size needed to do radix sorting, this can use uninitialized data.
    cub::DeviceRadixSort::SortPairs(
        d_sort_storage_,
        d_sort_storage_bytes_,
        d_sort_keys_in_,
        d_sort_keys_out_,
        d_sort_vals_in_,
        d_perm_,
        N_
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sort_storage_, d_sort_storage_bytes_));

    // Calculate size needed to perform compaction of tiles
    cub::DevicePartition::Flagged(
        d_partition_storage_,
        d_partition_storage_bytes_,
        d_tile_idxs_,
        d_default_tile_mask_,
        d_nblist_tiles_,
        d_compacted_ixn_count_,
        MAX_TILES
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_partition_storage_, d_partition_storage_bytes_));

};

template <typename RealType, bool Interpolated>
Nonbonded<RealType, Interpolated>::~Nonbonded() {

    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_perm_)); // nullptr if we never built nblist

    gpuErrchk(cudaFree(d_bin_to_idx_));
    gpuErrchk(cudaFree(d_sorted_x_));

    gpuErrchk(cudaFree(d_w_));
    gpuErrchk(cudaFree(d_dw_dl_));
    gpuErrchk(cudaFree(d_sorted_w_));
    gpuErrchk(cudaFree(d_sorted_dw_dl_));
    gpuErrchk(cudaFree(d_unsorted_p_));
    gpuErrchk(cudaFree(d_sorted_p_));
    gpuErrchk(cudaFree(d_unsorted_dp_dl_));
    gpuErrchk(cudaFree(d_sorted_dp_dl_));
    gpuErrchk(cudaFree(d_sorted_du_dx_));
    gpuErrchk(cudaFree(d_sorted_du_dp_));

    gpuErrchk(cudaFree(d_sort_keys_in_));
    gpuErrchk(cudaFree(d_sort_keys_out_));
    gpuErrchk(cudaFree(d_sort_vals_in_));
    gpuErrchk(cudaFree(d_sort_storage_));

    gpuErrchk(cudaFree(d_nblist_x_));
    gpuErrchk(cudaFree(d_nblist_box_));
    gpuErrchk(cudaFree(d_rebuild_nblist_));

    gpuErrchk(cudaFree(d_alchemical_u_));
    gpuErrchk(cudaFree(d_vanilla_u_));
    gpuErrchk(cudaFree(d_alchemical_du_dx_));
    gpuErrchk(cudaFree(d_vanilla_du_dx_));
    gpuErrchk(cudaFree(d_alchemical_du_dp_));
    gpuErrchk(cudaFree(d_vanilla_du_dp_));
    gpuErrchk(cudaFree(d_alchemical_du_dl_));
    gpuErrchk(cudaFree(d_vanilla_du_dl_));
    gpuErrchk(cudaFree(d_run_vanilla_tiles_));

    gpuErrchk(cudaFree(d_partition_storage_));
    gpuErrchk(cudaFree(d_nblist_tiles_));
    gpuErrchk(cudaFree(d_default_tile_mask_));
    gpuErrchk(cudaFree(d_tile_mask_));
    gpuErrchk(cudaFree(d_compaction_mask_));
    gpuErrchk(cudaFree(d_x_last_));
    gpuErrchk(cudaFree(d_p_last_));
    gpuErrchk(cudaFree(d_tile_idxs_));
};


template<typename RealType, bool Interpolated>
void Nonbonded<RealType, Interpolated>::set_nblist_padding(double val) {
    nblist_padding_ = val;
}


template<typename RealType, bool Interpolated>
void Nonbonded<RealType, Interpolated>::disable_hilbert_sort() {
    disable_hilbert_ = true;
}

template <typename RealType, bool Interpolated>
void Nonbonded<RealType, Interpolated>::hilbert_sort(
    const double *d_coords,
    const double *d_box,
    cudaStream_t stream) {

    const int tpb = 32;
    const int B = (N_+tpb-1)/tpb;

    k_coords_to_kv<<<B, tpb, 0, stream>>>(N_, d_coords, d_box, d_bin_to_idx_, d_sort_keys_in_, d_sort_vals_in_);

    gpuErrchk(cudaPeekAtLastError());

    cub::DeviceRadixSort::SortPairs(
        d_sort_storage_,
        d_sort_storage_bytes_,
        d_sort_keys_in_,
        d_sort_keys_out_,
        d_sort_vals_in_,
        d_perm_,
        N_,
        0, // begin bit
        sizeof(*d_sort_keys_in_)*8, // end bit
        stream // cudaStream
    );

    gpuErrchk(cudaPeekAtLastError());

}

template <typename RealType, bool Interpolated>
void Nonbonded<RealType, Interpolated>::execute_device(
    const int N,
    const int P,
    const double *d_x, // [N * 3]
    const double *d_p, // [N * 3]
    const double *d_box, // [3 * 3]
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    // (ytz) the nonbonded algorithm proceeds as follows:

    // (done in constructor), construct a hilbert curve mapping each of the 256x256x256 cells into an index.
    // a. decide if we need to rebuild the neighborlist, if so:
    //     - look up which cell each particle belongs to, and its linear index along the hilbert curve.
    //     - use radix pair sort keyed on the hilbert index with values equal to the atomic index
    //     - resulting sorted values is the permutation array.
    //     - permute lambda plane/offsets, coords
    // b. else:
    //     - permute new coords
    // c. permute parameters
    // d. compute the nonbonded interactions using the neighborlist
    // e. inverse permute the forces, du/dps into the original index.
    // f. u and du/dl is buffered into a per-particle array, and then reduced.
    // g. note that du/dl is not an exact per-particle du/dl - it is only used for reduction purposes.

    if(N != N_) {
        std::cout << N << " " << N_ << std::endl;
        throw std::runtime_error("Nonbonded::execute_device() N != N_");
    }
    bool sort_indices = cur_step_ == 0;
    cur_step_ = (cur_step_ + 1) % HILBERT_SORT_INTERVAL; // How often to sort
    const int M = Interpolated ? 2 : 1;

    if(P != M*N_*3) {
        std::cout << P << " " << N_ << std::endl;
        throw std::runtime_error("Nonbonded::execute_device() P != M*N_*3");
    }

    // identify which tiles contain interpolated parameters

    const int tpb = 32;
    const int B = (N+tpb-1)/tpb;
    gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 0, sizeof(*d_rebuild_nblist_), stream));
    dim3 dimGrid(B, 3, 1);
    // If we have to sort, we also have to rebuild the neighborlist
    if (sort_indices) {
        if(!disable_hilbert_) {
            this->hilbert_sort(d_x, d_box, stream);
        } else {
            k_arange<<<B, tpb, 0, stream>>>(N, d_perm_);
            gpuErrchk(cudaPeekAtLastError());
        }
        // Indicate that the neighborlist must be rebuilt
        gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 1, 1*sizeof(*d_rebuild_nblist_), stream));
        k_permute<<<B, tpb, 0, stream>>>(N, d_perm_, d_lambda_offset_idxs_, d_lambda_offset_idxs_sorted_);
        gpuErrchk(cudaPeekAtLastError());
        k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_x, d_sorted_x_);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_x, d_sorted_x_);
        gpuErrchk(cudaPeekAtLastError());
        // (ytz) see if we need to rebuild the neighborlist.
        // (ytz + jfass): note that this logic could be optimized when using NPT is
        // enabled since a resize in the box can introduce new interactions.
        k_check_rebuild_coords_and_box<RealType><<<B, tpb, 0, stream>>>(
            N,
            d_sorted_x_,
            d_nblist_x_,
            d_box,
            d_nblist_box_,
            nblist_padding_,
            d_rebuild_nblist_
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    // compute new coordinates, new lambda_idxs, new_plane_idxs
    nblist_.build_nblist_device(
        N,
        d_sorted_x_,
        d_box,
        cutoff_+nblist_padding_,
        d_rebuild_nblist_,
        stream
    );
    gpuErrchk(cudaPeekAtLastError());
    const unsigned int MAX_TILES = nblist_.B()*nblist_.B();
    const int X = (MAX_TILES + tpb - 1) / tpb;
    // Every time we rebuild the neighborlist, need to create a new 'default' mask that is used
    // in compacting the tiles that have ixns. At the same time, zero out the alchemical tile list
    k_update_default_masks<<<X, tpb, 0, stream>>>(MAX_TILES, d_rebuild_nblist_, nblist_.get_ixn_count(), d_default_tile_mask_, d_tile_mask_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpyAsync(d_nblist_box_, d_box, 3*3*sizeof(*d_box), cudaMemcpyDeviceToDevice, stream));

    // do parameter interpolation here
    if(Interpolated) {
        CUresult result = compute_permute_interpolated_.configure(dimGrid, tpb, 0, stream)
        .launch(
            lambda,
            N,
            d_perm_,
            d_p,
            d_sorted_p_,
            d_sorted_dp_dl_
        );
        if(result != 0) {
            throw std::runtime_error("Driver call to k_permute_interpolated failed");
        }
    } else {
        k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_p, d_sorted_p_);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemsetAsync(d_sorted_dp_dl_, 0, N*3*sizeof(*d_sorted_dp_dl_), stream))
    }


    // reset buffers and sorted accumulators
    if(d_du_dx) {
	    gpuErrchk(cudaMemsetAsync(d_sorted_du_dx_, 0, N*3*sizeof(*d_sorted_du_dx_), stream))
    }
    if(d_du_dp) {
	    gpuErrchk(cudaMemsetAsync(d_sorted_du_dp_, 0, N*3*sizeof(*d_sorted_du_dp_), stream))
    }

    // update new w coordinates
    // (tbd): cache lambda value for equilibrium calculations
    CUresult result = compute_w_coords_instance_.configure(B, tpb, 0, stream)
    .launch(
        N,
        lambda,
        cutoff_,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_w_,
        d_dw_dl_
    );
    if(result != 0) {
        throw std::runtime_error("Driver call to k_compute_w_coords");
    }

    gpuErrchk(cudaPeekAtLastError());
    k_permute_2x<<<B, tpb, 0, stream>>>(N, d_perm_, d_w_, d_dw_dl_, d_sorted_w_, d_sorted_dw_dl_);
    gpuErrchk(cudaPeekAtLastError());
    // look up which kernel we need for this computation
    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dl ? 1 << 1 : 0;
    kernel_idx |= d_du_dx ? 1 << 2 : 0;
    kernel_idx |= d_u ? 1 << 3 : 0;
    // We can only reuse cached portions if we run with the same kernel flags, else have to rest the caches
    if (last_kernel_ != kernel_idx) {
        last_kernel_ = kernel_idx;
        gpuErrchk(cudaMemsetAsync(d_run_vanilla_tiles_, 1, 1*sizeof(*d_run_vanilla_tiles_), stream));
        // gpuErrchk(cudaMemsetAsync(d_x_last_, 0, N*3*sizeof(*d_x_last_), stream));
        // Don't have to memset the whole array, since we check parity across all array
        gpuErrchk(cudaMemsetAsync(d_p_last_, 0, N*3*sizeof(*d_p_last_), stream));
    } else {
        gpuErrchk(cudaMemsetAsync(d_run_vanilla_tiles_, 0, 1*sizeof(*d_run_vanilla_tiles_), stream));
        k_check_rebuild_tiles_mask<<<B, tpb, 0, stream>>>(
            N,
            d_run_vanilla_tiles_, // [1] Rebuild tile mask
            d_box,
            d_sorted_x_,
            d_sorted_p_,
            d_nblist_x_,
            d_p_last_,
            d_nblist_box_
        );
        gpuErrchk(cudaPeekAtLastError());
        // Store the last d_x/d_p
        // gpuErrchk(cudaMemcpyAsync(d_x_last_, d_sorted_x_, N*3*sizeof(*d_x_last_), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_p_last_, d_sorted_p_, N*3*sizeof(*d_p_last_), cudaMemcpyDeviceToDevice, stream));
    }
    gpuErrchk(cudaMemcpyAsync(d_nblist_x_, d_sorted_x_, N*3*sizeof(*d_sorted_x_), cudaMemcpyDeviceToDevice, stream));
    if (d_u) {
        k_reset_buffers<<<B, tpb, 0, stream>>>(N, d_run_vanilla_tiles_, d_alchemical_u_, d_vanilla_u_);
        gpuErrchk(cudaPeekAtLastError());
    }
    if (d_du_dl) {
        k_reset_buffers<<<B, tpb, 0, stream>>>(N, d_run_vanilla_tiles_, d_alchemical_du_dl_, d_vanilla_du_dl_);
        gpuErrchk(cudaPeekAtLastError());
    }
    if (d_du_dx) {
        k_reset_buffers<<<dimGrid, tpb, 0, stream>>>(N, d_run_vanilla_tiles_, d_alchemical_du_dx_, d_vanilla_du_dx_);
        gpuErrchk(cudaPeekAtLastError());
    }

    if (d_du_dp) {
        k_reset_buffers<<<dimGrid, tpb, 0, stream>>>(N, d_run_vanilla_tiles_, d_alchemical_du_dp_, d_vanilla_du_dp_);
        gpuErrchk(cudaPeekAtLastError());
    }
    k_generate_alchemical_tile_mask<RealType, Interpolated><<<NONBONDED_KERNEL_BLOCKS, tpb, 0, stream>>>(
        N,
        d_rebuild_nblist_,
        nblist_.get_ixn_count(),
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_sorted_dp_dl_,
        d_lambda_offset_idxs_sorted_,
        d_tile_mask_
    );
    gpuErrchk(cudaPeekAtLastError());
    // If we are only running the alchemical, swap in the alchemical tile mask for the default
    k_optional_copy_mask<<<X, tpb, 0, stream>>>(MAX_TILES, d_run_vanilla_tiles_, d_default_tile_mask_, d_tile_mask_, d_compaction_mask_);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemsetAsync(d_compacted_ixn_count_, 0, 1*sizeof(*d_compacted_ixn_count_), stream));
    gpuErrchk(cudaMemsetAsync(d_nblist_tiles_, 0, MAX_TILES*sizeof(*d_nblist_tiles_), stream));
    cub::DevicePartition::Flagged(
        d_partition_storage_,
        d_partition_storage_bytes_,
        d_tile_idxs_,
        d_compaction_mask_,
        d_nblist_tiles_,
        d_compacted_ixn_count_,
        MAX_TILES,
        stream
    );
    gpuErrchk(cudaPeekAtLastError());

    kernel_ptrs_[kernel_idx]<<<NONBONDED_KERNEL_BLOCKS, tpb, 0, stream>>>(
        N,
        d_compacted_ixn_count_,
        d_nblist_tiles_,
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_sorted_x_,
        d_sorted_p_,
        d_box,
        d_sorted_dp_dl_,
        d_sorted_w_,
        d_sorted_dw_dl_,
        lambda,
        beta_,
        cutoff_,
        d_tile_mask_,
        d_alchemical_du_dx_,
        d_vanilla_du_dx_,
        d_alchemical_du_dp_,
        d_vanilla_du_dp_,
        d_alchemical_du_dl_,
        d_vanilla_du_dl_, // switch to nullptr if we don't request du_dl
        d_alchemical_u_, // Buffer to store alchemical energies
        d_vanilla_u_,
        d_run_vanilla_tiles_
    );

    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
        // If a rebuild of the energy cache is necessary, copy current energy to cache.
        k_add_arrays<<<B, tpb, 0, stream>>>(N, d_alchemical_u_, d_vanilla_u_, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }

    if (d_du_dl) {
        // If a rebuild of the energy cache is necessary, copy current energy to cache.
        k_add_arrays<<<B, tpb, 0, stream>>>(N, d_alchemical_du_dl_, d_vanilla_du_dl_, d_du_dl);
        gpuErrchk(cudaPeekAtLastError());
    }

    // coords are N,3
    if(d_du_dx) {
        k_add_arrays<<<dimGrid, tpb, 0, stream>>>(N, d_alchemical_du_dx_, d_vanilla_du_dx_, d_sorted_du_dx_);
        gpuErrchk(cudaPeekAtLastError());
        k_inv_permute_accum<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_sorted_du_dx_, d_du_dx);
        gpuErrchk(cudaPeekAtLastError());
    }

    // params are N,3
    // this needs to be an accumulated permute
    if(d_du_dp) {
        k_add_arrays<<<dimGrid, tpb, 0, stream>>>(N, d_alchemical_du_dp_, d_vanilla_du_dp_, d_sorted_du_dp_);
        gpuErrchk(cudaPeekAtLastError());
        k_inv_permute_assign<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_sorted_du_dp_, d_du_dp_buffer_);
        gpuErrchk(cudaPeekAtLastError());
    }

    // exclusions use the non-sorted version
    if(E_ > 0) {

        dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

        if(Interpolated) {
            k_inv_permute_assign_2x<<<dimGrid, tpb, 0, stream>>>(
                N,
                d_perm_,
                d_sorted_p_,
                d_sorted_dp_dl_,
                d_unsorted_p_,
                d_unsorted_dp_dl_
            );
            gpuErrchk(cudaPeekAtLastError());
        }

        k_nonbonded_exclusions<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_x,
            Interpolated ? d_unsorted_p_ : d_p,
            d_box,
            Interpolated ? d_unsorted_dp_dl_ : d_sorted_dp_dl_,
            d_w_,
            d_dw_dl_,
            lambda,
            d_exclusion_idxs_,
            d_scales_,
            beta_,
            cutoff_,
            d_du_dx,
            d_du_dp_buffer_,
            d_du_dl,
            d_u
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_du_dp) {
        if(Interpolated) {
            CUresult result = compute_add_ull_to_real_interpolated_.configure(dimGrid, tpb, 0, stream)
            .launch(
                lambda,
                N,
                d_du_dp_buffer_,
                d_du_dp
            );
            if(result != 0) {
                throw std::runtime_error("Driver call to k_add_ull_to_real_interpolated failed");
            }
        } else {
            k_add_ull_to_real<<<dimGrid, tpb, 0, stream>>>(
                N,
                d_du_dp_buffer_,
                d_du_dp
            );
        }
        gpuErrchk(cudaPeekAtLastError());
    }

}

template class Nonbonded<double, true>;
template class Nonbonded<float, true>;
template class Nonbonded<double, false>;
template class Nonbonded<float, false>;


} // namespace timemachine
