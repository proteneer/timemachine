#include <cub/cub.cuh>
#include <string>

#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_common.cuh"
#include "vendored/hilbert.h"

#include "k_lambda_transformer.cuh"
#include "k_nonbonded.cuh"

namespace timemachine {

template <typename RealType>
NonbondedAllPairs<RealType>::NonbondedAllPairs(
    const std::vector<int> &lambda_plane_idxs,  // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    const double beta,
    const double cutoff,
    const std::optional<std::set<int>> &atom_idxs)
    : N_(lambda_offset_idxs.size()), K_(atom_idxs ? atom_idxs->size() : N_), beta_(beta), cutoff_(cutoff),
      d_atom_idxs_(nullptr), nblist_(K_), nblist_padding_(0.1), d_sort_storage_(nullptr), d_sort_storage_bytes_(0),
      disable_hilbert_(false),

      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP
                    //                             U  X  P
                    &k_nonbonded_unified<RealType, 0, 0, 0>,
                    &k_nonbonded_unified<RealType, 0, 0, 1>,
                    &k_nonbonded_unified<RealType, 0, 1, 0>,
                    &k_nonbonded_unified<RealType, 0, 1, 1>,
                    &k_nonbonded_unified<RealType, 1, 0, 0>,
                    &k_nonbonded_unified<RealType, 1, 0, 1>,
                    &k_nonbonded_unified<RealType, 1, 1, 0>,
                    &k_nonbonded_unified<RealType, 1, 1, 1>}) {

    if (lambda_offset_idxs.size() != lambda_plane_idxs.size()) {
        throw std::runtime_error("lambda offset idxs and plane idxs need to be equivalent");
    }

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_ * sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(
        d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_ * sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_ * sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(
        d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_ * sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    if (atom_idxs) {
        gpuErrchk(cudaMalloc(&d_atom_idxs_, K_ * sizeof(*d_atom_idxs_)));
        std::vector<int> atom_idxs_v(atom_idxs->begin(), atom_idxs->end());
        gpuErrchk(cudaMemcpy(d_atom_idxs_, &atom_idxs_v[0], K_ * sizeof(*d_atom_idxs_), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMalloc(&d_sorted_atom_idxs_, K_ * sizeof(*d_sorted_atom_idxs_)));

    gpuErrchk(cudaMalloc(&d_gathered_x_, K_ * 3 * sizeof(*d_gathered_x_)));

    gpuErrchk(cudaMalloc(&d_w_, N_ * sizeof(*d_w_)));

    gpuErrchk(cudaMalloc(&d_gathered_w_, K_ * sizeof(*d_gathered_w_)));

    gpuErrchk(cudaMalloc(&d_gathered_p_, K_ * PARAMS_PER_ATOM * sizeof(*d_gathered_p_)));
    gpuErrchk(cudaMalloc(&d_gathered_du_dx_, K_ * 3 * sizeof(*d_gathered_du_dx_)));
    gpuErrchk(cudaMalloc(&d_gathered_du_dp_, K_ * PARAMS_PER_ATOM * sizeof(*d_gathered_du_dp_)));

    gpuErrchk(cudaMalloc(&d_du_dp_buffer_, N_ * PARAMS_PER_ATOM * sizeof(*d_du_dp_buffer_)));

    gpuErrchk(cudaMallocHost(&p_ixn_count_, 1 * sizeof(*p_ixn_count_)));

    gpuErrchk(cudaMalloc(&d_nblist_x_, N_ * 3 * sizeof(*d_nblist_x_)));
    gpuErrchk(cudaMemset(d_nblist_x_, 0, N_ * 3 * sizeof(*d_nblist_x_))); // set non-sensical positions
    gpuErrchk(cudaMalloc(&d_nblist_box_, 3 * 3 * sizeof(*d_nblist_x_)));
    gpuErrchk(cudaMemset(d_nblist_box_, 0, 3 * 3 * sizeof(*d_nblist_x_)));
    gpuErrchk(cudaMalloc(&d_rebuild_nblist_, 1 * sizeof(*d_rebuild_nblist_)));
    gpuErrchk(cudaMallocHost(&p_rebuild_nblist_, 1 * sizeof(*p_rebuild_nblist_)));

    gpuErrchk(cudaMalloc(&d_sort_keys_in_, K_ * sizeof(d_sort_keys_in_)));
    gpuErrchk(cudaMalloc(&d_sort_keys_out_, K_ * sizeof(d_sort_keys_out_)));
    gpuErrchk(cudaMalloc(&d_sort_vals_in_, K_ * sizeof(d_sort_vals_in_)));

    // initialize hilbert curve
    std::vector<unsigned int> bin_to_idx(HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM);
    for (int i = 0; i < HILBERT_GRID_DIM; i++) {
        for (int j = 0; j < HILBERT_GRID_DIM; j++) {
            for (int k = 0; k < HILBERT_GRID_DIM; k++) {

                bitmask_t hilbert_coords[3];
                hilbert_coords[0] = i;
                hilbert_coords[1] = j;
                hilbert_coords[2] = k;

                unsigned int bin = static_cast<unsigned int>(hilbert_c2i(3, HILBERT_N_BITS, hilbert_coords));
                bin_to_idx[i * HILBERT_GRID_DIM * HILBERT_GRID_DIM + j * HILBERT_GRID_DIM + k] = bin;
            }
        }
    }

    gpuErrchk(
        cudaMalloc(&d_bin_to_idx_, HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM * sizeof(*d_bin_to_idx_)));
    gpuErrchk(cudaMemcpy(
        d_bin_to_idx_,
        &bin_to_idx[0],
        HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM * sizeof(*d_bin_to_idx_),
        cudaMemcpyHostToDevice));

    // estimate size needed to do radix sorting, this can use uninitialized data.
    cub::DeviceRadixSort::SortPairs(
        nullptr, d_sort_storage_bytes_, d_sort_keys_in_, d_sort_keys_out_, d_sort_vals_in_, d_sorted_atom_idxs_, K_);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sort_storage_, d_sort_storage_bytes_));
};

template <typename RealType> NonbondedAllPairs<RealType>::~NonbondedAllPairs() {

    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

    if (d_atom_idxs_) {
        gpuErrchk(cudaFree(d_atom_idxs_));
    }

    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_sorted_atom_idxs_));

    gpuErrchk(cudaFree(d_bin_to_idx_));
    gpuErrchk(cudaFree(d_gathered_x_));

    gpuErrchk(cudaFree(d_w_));
    gpuErrchk(cudaFree(d_gathered_w_));
    gpuErrchk(cudaFree(d_gathered_p_));
    gpuErrchk(cudaFree(d_gathered_du_dx_));
    gpuErrchk(cudaFree(d_gathered_du_dp_));

    gpuErrchk(cudaFree(d_sort_keys_in_));
    gpuErrchk(cudaFree(d_sort_keys_out_));
    gpuErrchk(cudaFree(d_sort_vals_in_));
    gpuErrchk(cudaFree(d_sort_storage_));

    gpuErrchk(cudaFreeHost(p_ixn_count_));

    gpuErrchk(cudaFree(d_nblist_x_));
    gpuErrchk(cudaFree(d_nblist_box_));
    gpuErrchk(cudaFree(d_rebuild_nblist_));
    gpuErrchk(cudaFreeHost(p_rebuild_nblist_));
};

template <typename RealType> void NonbondedAllPairs<RealType>::set_nblist_padding(double val) { nblist_padding_ = val; }

template <typename RealType> void NonbondedAllPairs<RealType>::disable_hilbert_sort() { disable_hilbert_ = true; }

template <typename RealType>
void NonbondedAllPairs<RealType>::hilbert_sort(const double *d_coords, const double *d_box, cudaStream_t stream) {

    const int tpb = 32;
    const int B = ceil_divide(K_, tpb);

    if (d_atom_idxs_) {
        k_coords_to_kv_gather<<<B, tpb, 0, stream>>>(
            K_, d_atom_idxs_, d_coords, d_box, d_bin_to_idx_, d_sort_keys_in_, d_sort_vals_in_);
    } else {
        // N_ == K_
        k_coords_to_kv<<<B, tpb, 0, stream>>>(K_, d_coords, d_box, d_bin_to_idx_, d_sort_keys_in_, d_sort_vals_in_);
    }

    gpuErrchk(cudaPeekAtLastError());

    cub::DeviceRadixSort::SortPairs(
        d_sort_storage_,
        d_sort_storage_bytes_,
        d_sort_keys_in_,
        d_sort_keys_out_,
        d_sort_vals_in_,
        d_sorted_atom_idxs_,
        K_,
        0,                            // begin bit
        sizeof(*d_sort_keys_in_) * 8, // end bit
        stream                        // cudaStream
    );

    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void NonbondedAllPairs<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,   // N * PARAMS_PER_ATOM
    const double *d_box, // 3 * 3
    const double lambda,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    // (ytz) the nonbonded algorithm proceeds as follows:

    // (done in constructor), construct a hilbert curve mapping each of the HILBERT_GRID_DIM x HILBERT_GRID_DIM x HILBERT_GRID_DIM cells into an index.
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

    if (N != N_) {
        throw std::runtime_error(
            "NonbondedAllPairs::execute_device(): expected N == N_, got N=" + std::to_string(N) +
            ", N_=" + std::to_string(N_));
    }

    if (P != N_ * PARAMS_PER_ATOM) {
        throw std::runtime_error(
            "NonbondedAllPairs::execute_device(): expected P == N_*" + std::to_string(PARAMS_PER_ATOM) + ", got P=" +
            std::to_string(P) + ", N_*" + std::to_string(PARAMS_PER_ATOM) + "=" + std::to_string(N_ * PARAMS_PER_ATOM));
    }

    const int tpb = 32;

    // (ytz) see if we need to rebuild the neighborlist.
    if (d_atom_idxs_) {
        k_check_rebuild_coords_and_box_gather<RealType><<<ceil_divide(K_, tpb), tpb, 0, stream>>>(
            K_, d_atom_idxs_, d_x, d_nblist_x_, d_box, d_nblist_box_, nblist_padding_, d_rebuild_nblist_);
    } else {
        k_check_rebuild_coords_and_box<RealType><<<ceil_divide(K_, tpb), tpb, 0, stream>>>(
            K_, d_x, d_nblist_x_, d_box, d_nblist_box_, nblist_padding_, d_rebuild_nblist_);
    }
    gpuErrchk(cudaPeekAtLastError());

    // we can optimize this away by doing the check on the GPU directly.
    gpuErrchk(cudaMemcpyAsync(
        p_rebuild_nblist_, d_rebuild_nblist_, 1 * sizeof(*p_rebuild_nblist_), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream)); // slow!

    if (p_rebuild_nblist_[0] > 0) {

        // (ytz): update the permutation index before building neighborlist, as the neighborlist is tied
        // to a particular sort order
        if (!disable_hilbert_) {
            this->hilbert_sort(d_x, d_box, stream);
        } else {
            if (d_atom_idxs_) {
                gpuErrchk(cudaMemcpyAsync(
                    d_sorted_atom_idxs_, d_atom_idxs_, K_ * sizeof(*d_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
                gpuErrchk(cudaPeekAtLastError());
            } else {
                // N_ == K_
                k_arange<<<ceil_divide(K_, tpb), tpb, 0, stream>>>(K_, d_sorted_atom_idxs_);
            }
        }

        // compute new coordinates, new lambda_idxs, new_plane_idxs
        k_gather<<<dim3(ceil_divide(K_, tpb), 3, 1), tpb, 0, stream>>>(K_, d_sorted_atom_idxs_, d_x, d_gathered_x_);
        gpuErrchk(cudaPeekAtLastError());
        nblist_.build_nblist_device(K_, d_gathered_x_, d_box, cutoff_ + nblist_padding_, stream);
        gpuErrchk(cudaMemcpyAsync(
            p_ixn_count_, nblist_.get_ixn_count(), 1 * sizeof(*p_ixn_count_), cudaMemcpyDeviceToHost, stream));

        std::vector<double> h_box(9);
        gpuErrchk(cudaMemcpyAsync(&h_box[0], d_box, 3 * 3 * sizeof(*d_box), cudaMemcpyDeviceToHost, stream));

        // this stream needs to be synchronized so we can be sure that p_ixn_count_ is properly set.
        gpuErrchk(cudaStreamSynchronize(stream));

        // Verify that the cutoff and box size are valid together. If cutoff is greater than half the box
        // then a particle can interact with multiple periodic copies.
        const double db_cutoff = (cutoff_ + nblist_padding_) * 2;

        // Verify that box is orthogonal and the width of the box in all dimensions is greater than twice the cutoff
        for (int i = 0; i < 9; i++) {
            if (i == 0 || i == 4 || i == 8) {
                if (h_box[i] < db_cutoff) {
                    throw std::runtime_error(
                        "Cutoff with padding is more than half of the box width, neighborlist is no longer reliable");
                }
            } else if (h_box[i] != 0.0) {
                throw std::runtime_error("Provided non-ortholinear box, unable to compute nonbonded energy");
            }
        }

        gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 0, sizeof(*d_rebuild_nblist_), stream));
        gpuErrchk(cudaMemcpyAsync(d_nblist_x_, d_x, N * 3 * sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_nblist_box_, d_box, 3 * 3 * sizeof(*d_box), cudaMemcpyDeviceToDevice, stream));
    } else {
        k_gather<<<dim3(ceil_divide(K_, tpb), 3, 1), tpb, 0, stream>>>(K_, d_sorted_atom_idxs_, d_x, d_gathered_x_);
        gpuErrchk(cudaPeekAtLastError());
    }

    // do parameter interpolation here
    k_gather<<<dim3(ceil_divide(K_, tpb), PARAMS_PER_ATOM, 1), tpb, 0, stream>>>(
        K_, d_sorted_atom_idxs_, d_p, d_gathered_p_);
    gpuErrchk(cudaPeekAtLastError());

    // reset buffers and sorted accumulators
    if (d_du_dx) {
        gpuErrchk(cudaMemsetAsync(d_gathered_du_dx_, 0, K_ * 3 * sizeof(*d_gathered_du_dx_), stream))
    }
    if (d_du_dp) {
        gpuErrchk(cudaMemsetAsync(d_gathered_du_dp_, 0, K_ * PARAMS_PER_ATOM * sizeof(*d_gathered_du_dp_), stream))
    }

    // update new w coordinates
    // (tbd): cache lambda value for equilibrium calculations
    k_compute_w_coords<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
        N, lambda, cutoff_, d_lambda_plane_idxs_, d_lambda_offset_idxs_, d_w_);
    gpuErrchk(cudaPeekAtLastError());

    k_gather<<<ceil_divide(K_, tpb), tpb, 0, stream>>>(K_, d_sorted_atom_idxs_, d_w_, d_gathered_w_);
    gpuErrchk(cudaPeekAtLastError());

    // look up which kernel we need for this computation
    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<p_ixn_count_[0], tpb, 0, stream>>>(
        K_,
        nblist_.get_num_row_idxs(),
        d_gathered_x_,
        d_gathered_p_,
        d_box,
        d_gathered_w_,
        beta_,
        cutoff_,
        nblist_.get_row_idxs(),
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_gathered_du_dx_,
        d_gathered_du_dp_,
        d_u // switch to nullptr if we don't request energies
    );

    gpuErrchk(cudaPeekAtLastError());

    // coords are N,3
    if (d_du_dx) {
        k_scatter_accum<<<dim3(ceil_divide(K_, tpb), 3, 1), tpb, 0, stream>>>(
            K_, d_sorted_atom_idxs_, d_gathered_du_dx_, d_du_dx);
        gpuErrchk(cudaPeekAtLastError());
    }

    // params are N, PARAMS_PER_ATOM
    // this needs to be an accumulated permute
    if (d_du_dp) {
        // scattered assignment updates K_ <= N_ elements; the rest should be 0
        gpuErrchk(cudaMemsetAsync(d_du_dp_buffer_, 0, N_ * PARAMS_PER_ATOM * sizeof(*d_du_dp_buffer_), stream));
        k_scatter_assign<<<dim3(ceil_divide(K_, tpb), PARAMS_PER_ATOM, 1), tpb, 0, stream>>>(
            K_, d_sorted_atom_idxs_, d_gathered_du_dp_, d_du_dp_buffer_);
        gpuErrchk(cudaPeekAtLastError());
    }

    if (d_du_dp) {
        k_add_ull_to_ull<<<dim3(ceil_divide(N_, tpb), PARAMS_PER_ATOM, 1), tpb, 0, stream>>>(
            N, d_du_dp_buffer_, d_du_dp);
        gpuErrchk(cudaPeekAtLastError());
    }
}

template <typename RealType>
void NonbondedAllPairs<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    for (int i = 0; i < N; i++) {
        const int offset = i * PARAMS_PER_ATOM;
        const int idx_charge = offset + PARAM_OFFSET_CHARGE;
        const int idx_sig = offset + PARAM_OFFSET_SIG;
        const int idx_eps = offset + PARAM_OFFSET_EPS;

        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
    }
}

template class NonbondedAllPairs<double>;
template class NonbondedAllPairs<float>;

} // namespace timemachine
