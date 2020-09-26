#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cub/cub.cuh>

#include "nonbonded.hpp"
#include "hilbert.h"
#include "gpu_utils.cuh"

#include "k_nonbonded.cuh"

namespace timemachine {

template <typename RealType>
Nonbonded<RealType>::Nonbonded(
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &scales, // [E, 2]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double beta,
    double cutoff
) :  N_(lambda_offset_idxs.size()),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_offset_idxs.size()),
    beta_(beta),
    d_sort_storage_(nullptr),
    d_sort_storage_bytes_(0) {

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(scales.size()/2 != E_) {
        throw std::runtime_error("bad scales size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, N_*sizeof(*d_du_dl_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, N_*sizeof(*d_u_buffer_)));
    gpuErrchk(cudaMalloc(&d_perm_, N_*sizeof(*d_perm_)));

    gpuErrchk(cudaMalloc(&d_sorted_lambda_offset_idxs_, N_*sizeof(*d_sorted_lambda_offset_idxs_)));
    gpuErrchk(cudaMalloc(&d_sorted_x_, N_*3*sizeof(*d_sorted_x_)));
    gpuErrchk(cudaMalloc(&d_sorted_p_, N_*3*sizeof(*d_sorted_p_)));
    gpuErrchk(cudaMalloc(&d_sorted_du_dx_, N_*3*sizeof(*d_sorted_du_dx_)));
    gpuErrchk(cudaMalloc(&d_sorted_du_dp_, N_*3*sizeof(*d_sorted_du_dp_)));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_scales_, E_*2*sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], E_*2*sizeof(*d_scales_), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMallocHost(&p_ixn_count_, 1*sizeof(*p_ixn_count_)));

    gpuErrchk(cudaMalloc(&d_sort_keys_in_, N_*sizeof(d_sort_keys_in_)));
    gpuErrchk(cudaMalloc(&d_sort_keys_out_, N_*sizeof(d_sort_keys_out_)));
    gpuErrchk(cudaMalloc(&d_sort_vals_in_, N_*sizeof(d_sort_vals_in_)));

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

};


template <typename RealType>
Nonbonded<RealType>::~Nonbonded() {

    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_perm_)); // nullptr if we never built nblist
    
    gpuErrchk(cudaFree(d_bin_to_idx_));
    gpuErrchk(cudaFree(d_sorted_x_));
    gpuErrchk(cudaFree(d_sorted_p_));
    gpuErrchk(cudaFree(d_sorted_du_dx_));
    gpuErrchk(cudaFree(d_sorted_du_dp_));
    gpuErrchk(cudaFree(d_sorted_lambda_offset_idxs_));

    gpuErrchk(cudaFree(d_sort_keys_in_));
    gpuErrchk(cudaFree(d_sort_keys_out_));
    gpuErrchk(cudaFree(d_sort_vals_in_));
    gpuErrchk(cudaFree(d_sort_storage_));

    gpuErrchk(cudaFreeHost(p_ixn_count_));
};


template <typename RealType>
void Nonbonded<RealType>::hilbert_sort(
    const double *d_coords,
    const double *d_box,
    cudaStream_t stream) {

    const int B = (N_+32-1)/32;
    const int tpb = 32;

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

template <typename RealType>
void Nonbonded<RealType>::execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p, // N*3
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream) {

    // (ytz) the nonbonded algorithm proceeds as follows:

    // 0. (done in constructor), construct a hilbert curve mapping each of the 256x256x256 cells into an index.
    // 1. look up which cell each particle belongs to, and its linear index along the hilbert curve.
    // 2. use radix pair sort keyed on the hilbert index with values equal to the atomic index
    // 3. resulting sorted values is the permutation array.
    // 4. permute coords, params, lambda_offsets
    // 5. compute the neighborlist into tiles
    // 6. compute the nonbonded interactions using the neighborlist
    // 7. inverse permute the forces, du/dps into the original index.
    // 8. u and du/dl is buffered into a per-particle array, and then reduced.
    // 9. note that du/dl is not an exact per-particle du/dl - it is only used for reduction purposes.

    assert(N == N_);
    assert(P == N_*3);

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    const int B = (N+32-1)/32;
    const int tpb = 32;

    this->hilbert_sort(d_x, d_box, stream);

	dim3 dimGrid(B, 3, 1);

    k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_p, d_sorted_p_);
    gpuErrchk(cudaPeekAtLastError());

    k_permute<<<B, tpb, 0, stream>>>(N, d_perm_, d_lambda_offset_idxs_, d_sorted_lambda_offset_idxs_);
    gpuErrchk(cudaPeekAtLastError());

    k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_x, d_sorted_x_);
    gpuErrchk(cudaPeekAtLastError());


    nblist_.build_nblist_device(
        N,
        d_sorted_x_,
        d_box,
        cutoff_,
        stream
    );

    gpuErrchk(cudaMemcpyAsync(p_ixn_count_, nblist_.get_ixn_count(), 1*sizeof(*p_ixn_count_), cudaMemcpyDeviceToHost, stream));
    // this stream needs to be synchronized so we can be sure that p_ixn_count_ is properly set.
    gpuErrchk(cudaStreamSynchronize(stream));

    // reset buffers and sorted accumulators
    if(d_du_dx) {
	   gpuErrchk(cudaMemsetAsync(d_sorted_du_dx_, 0, N*3*sizeof(*d_sorted_du_dx_)))
    }
    if(d_du_dp) {
	   gpuErrchk(cudaMemsetAsync(d_sorted_du_dp_, 0, N*3*sizeof(*d_sorted_du_dp_)))
    }
    if(d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, N*sizeof(*d_du_dl_buffer_), stream));        
    }
    if(d_u) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N*sizeof(*d_du_dl_buffer_), stream));        
    }

    k_nonbonded<RealType><<<p_ixn_count_[0], 32, 0, stream>>>(
        N,
        d_sorted_x_,
        d_sorted_p_,
        d_box,
        lambda,
        d_sorted_lambda_offset_idxs_,
        beta_,
        cutoff_,
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_du_dx ? d_sorted_du_dx_ : nullptr,
        d_du_dp ? d_sorted_du_dp_ : nullptr,
        d_du_dl ? d_du_dl_buffer_ : nullptr, // switch to nullptr if we don't request du_dl
        d_u ? d_u_buffer_ : nullptr // switch to nullptr if we don't request energies
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // coords are N,3
    if(d_du_dx) {
        k_inv_permute_accum<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_sorted_du_dx_, d_du_dx);
        gpuErrchk(cudaPeekAtLastError());
    }

    // params are N,3
    // this needs to be an accumlated permute
    if(d_du_dp) {
        k_inv_permute_accum<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_sorted_du_dp_, d_du_dp);
        gpuErrchk(cudaPeekAtLastError());
    }

    // exclusions use the non-sorted version
    if(E_ > 0) {

        const int tpb = 32;
        dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

        k_nonbonded_exclusions<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_x,
            d_p,
            d_box,
            lambda,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_scales_,
            beta_,
            cutoff_,
            d_du_dx,
            d_du_dp,
            d_du_dl ? d_du_dl_buffer_ : nullptr, // switch to nullptr if we don't request du_dl
            d_u ? d_u_buffer_ : nullptr // switch to nullptr if we don't request energies
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_du_dl) {
        k_reduce_buffer<<<B, 32, 0, stream>>>(N, d_du_dl_buffer_, d_du_dl);
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_u) {
        k_reduce_buffer<<<B, 32, 0, stream>>>(N, d_u_buffer_, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
    
}

template class Nonbonded<double>;
template class Nonbonded<float>;

} // namespace timemachine
