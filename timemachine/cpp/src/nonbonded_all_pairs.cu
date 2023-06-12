#include <cub/cub.cuh>
#include <string>

#include "device_buffer.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_nonbonded.cuh"
#include "kernels/k_nonbonded_common.cuh"
#include "kernels/kernel_utils.cuh"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_common.hpp"
#include "vendored/hilbert.h"

#include <numeric>

static const int STEPS_PER_SORT = 100;

namespace timemachine {

template <typename RealType>
NonbondedAllPairs<RealType>::NonbondedAllPairs(
    const int N,
    const double beta,
    const double cutoff,
    const std::optional<std::set<int>> &atom_idxs,
    const bool disable_hilbert_sort,
    const double nblist_padding)
    : N_(N), K_(atom_idxs ? atom_idxs->size() : N_), beta_(beta), cutoff_(cutoff), steps_since_last_sort_(0),
      d_atom_idxs_(nullptr), nblist_(N_), nblist_padding_(nblist_padding), d_sort_storage_(nullptr),
      d_sort_storage_bytes_(0), disable_hilbert_(disable_hilbert_sort),

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

    std::vector<int> atom_idxs_h;
    if (atom_idxs) {
        atom_idxs_h = std::vector<int>(atom_idxs->begin(), atom_idxs->end());
    } else {
        atom_idxs_h = std::vector<int>(N_);
        std::iota(atom_idxs_h.begin(), atom_idxs_h.end(), 0);
    }
    verify_atom_idxs(N_, atom_idxs_h);

    cudaSafeMalloc(&d_atom_idxs_, N_ * sizeof(*d_atom_idxs_));

    cudaSafeMalloc(&d_sorted_atom_idxs_, N_ * sizeof(*d_sorted_atom_idxs_));

    cudaSafeMalloc(&d_gathered_x_, N_ * 3 * sizeof(*d_gathered_x_));

    cudaSafeMalloc(&d_gathered_p_, N_ * PARAMS_PER_ATOM * sizeof(*d_gathered_p_));
    cudaSafeMalloc(&d_gathered_du_dx_, N_ * 3 * sizeof(*d_gathered_du_dx_));
    cudaSafeMalloc(&d_gathered_du_dp_, N_ * PARAMS_PER_ATOM * sizeof(*d_gathered_du_dp_));

    cudaSafeMalloc(&d_du_dp_buffer_, N_ * PARAMS_PER_ATOM * sizeof(*d_du_dp_buffer_));

    gpuErrchk(cudaMallocHost(&p_ixn_count_, 1 * sizeof(*p_ixn_count_)));
    gpuErrchk(cudaMallocHost(&p_box_, 3 * 3 * sizeof(*p_box_)));

    cudaSafeMalloc(&d_nblist_x_, N_ * 3 * sizeof(*d_nblist_x_));
    gpuErrchk(cudaMemset(d_nblist_x_, 0, N_ * 3 * sizeof(*d_nblist_x_))); // set non-sensical positions
    cudaSafeMalloc(&d_nblist_box_, 3 * 3 * sizeof(*d_nblist_x_));
    gpuErrchk(cudaMemset(d_nblist_box_, 0, 3 * 3 * sizeof(*d_nblist_x_)));
    cudaSafeMalloc(&d_rebuild_nblist_, 1 * sizeof(*d_rebuild_nblist_));
    gpuErrchk(cudaMallocHost(&p_rebuild_nblist_, 1 * sizeof(*p_rebuild_nblist_)));

    cudaSafeMalloc(&d_sort_keys_in_, N_ * sizeof(d_sort_keys_in_));
    cudaSafeMalloc(&d_sort_keys_out_, N_ * sizeof(d_sort_keys_out_));
    cudaSafeMalloc(&d_sort_vals_in_, N_ * sizeof(d_sort_vals_in_));

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

    cudaSafeMalloc(&d_bin_to_idx_, HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM * sizeof(*d_bin_to_idx_));
    gpuErrchk(cudaMemcpy(
        d_bin_to_idx_,
        &bin_to_idx[0],
        HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM * sizeof(*d_bin_to_idx_),
        cudaMemcpyHostToDevice));

    // estimate size needed to do radix sorting, this can use uninitialized data.
    cub::DeviceRadixSort::SortPairs(
        nullptr, d_sort_storage_bytes_, d_sort_keys_in_, d_sort_keys_out_, d_sort_vals_in_, d_sorted_atom_idxs_, K_);

    gpuErrchk(cudaPeekAtLastError());

    cudaSafeMalloc(&d_sort_storage_, d_sort_storage_bytes_);

    this->set_atom_idxs(atom_idxs_h);

    // Create event with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&nblist_flag_sync_event_, cudaEventDisableTiming));
};

template <typename RealType> NonbondedAllPairs<RealType>::~NonbondedAllPairs() {

    gpuErrchk(cudaFree(d_atom_idxs_));

    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_sorted_atom_idxs_));

    gpuErrchk(cudaFree(d_bin_to_idx_));
    gpuErrchk(cudaFree(d_gathered_x_));

    gpuErrchk(cudaFree(d_gathered_p_));
    gpuErrchk(cudaFree(d_gathered_du_dx_));
    gpuErrchk(cudaFree(d_gathered_du_dp_));

    gpuErrchk(cudaFree(d_sort_keys_in_));
    gpuErrchk(cudaFree(d_sort_keys_out_));
    gpuErrchk(cudaFree(d_sort_vals_in_));
    gpuErrchk(cudaFree(d_sort_storage_));

    gpuErrchk(cudaFreeHost(p_ixn_count_));
    gpuErrchk(cudaFreeHost(p_box_));

    gpuErrchk(cudaFree(d_nblist_x_));
    gpuErrchk(cudaFree(d_nblist_box_));
    gpuErrchk(cudaFree(d_rebuild_nblist_));
    gpuErrchk(cudaFreeHost(p_rebuild_nblist_));

    gpuErrchk(cudaEventDestroy(nblist_flag_sync_event_));
};

// Set atom idxs upon which to compute the non-bonded potential. This will trigger a neighborlist rebuild.
template <typename RealType> void NonbondedAllPairs<RealType>::set_atom_idxs(const std::vector<int> &atom_idxs) {
    verify_atom_idxs(N_, atom_idxs);
    const cudaStream_t stream = static_cast<cudaStream_t>(0);
    std::vector<unsigned int> unsigned_idxs = std::vector<unsigned int>(atom_idxs.begin(), atom_idxs.end());
    DeviceBuffer<unsigned int> atom_idxs_buffer(atom_idxs.size());
    atom_idxs_buffer.copy_from(&unsigned_idxs[0]);
    this->set_atom_idxs_device(atom_idxs.size(), atom_idxs_buffer.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

template <typename RealType> std::vector<int> NonbondedAllPairs<RealType>::get_atom_idxs() {
    std::vector<unsigned int> atom_idxs_buffer(K_);
    gpuErrchk(cudaMemcpy(&atom_idxs_buffer[0], d_atom_idxs_, K_ * sizeof(*d_atom_idxs_), cudaMemcpyDeviceToHost));
    std::vector<int> atom_idxs = std::vector<int>(atom_idxs_buffer.begin(), atom_idxs_buffer.end());
    return atom_idxs;
}

template <typename RealType>
void NonbondedAllPairs<RealType>::set_atom_idxs_device(
    const int K, const unsigned int *d_in_atom_idxs, const cudaStream_t stream) {
    if (K < 1) {
        throw std::runtime_error("K must be at least 1");
    }
    if (K > N_) {
        throw std::runtime_error("number of idxs must be less than or equal to N");
    }
    gpuErrchk(
        cudaMemcpyAsync(d_atom_idxs_, d_in_atom_idxs, K * sizeof(*d_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
    nblist_.resize_device(K, stream);
    // Force the rebuild of the nblist
    gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 1, 1 * sizeof(*d_rebuild_nblist_), stream));
    this->K_ = K;
    // Reset the steps so that we do a new sort
    this->steps_since_last_sort_ = 0;
}

template <typename RealType> bool NonbondedAllPairs<RealType>::needs_sort() {
    return steps_since_last_sort_ % STEPS_PER_SORT == 0;
}

template <typename RealType>
void NonbondedAllPairs<RealType>::sort(const double *d_coords, const double *d_box, cudaStream_t stream) {
    // We must rebuild the neighborlist after sorting, as the neighborlist is tied to a particular sort order
    if (!disable_hilbert_) {
        this->hilbert_sort(d_coords, d_box, stream);
    } else {
        gpuErrchk(cudaMemcpyAsync(
            d_sorted_atom_idxs_, d_atom_idxs_, K_ * sizeof(*d_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
    }
    gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 1, sizeof(*d_rebuild_nblist_), stream));
    // Set the pinned memory to indicate that we need to rebuild
    p_rebuild_nblist_[0] = 1;
}

template <typename RealType>
void NonbondedAllPairs<RealType>::hilbert_sort(const double *d_coords, const double *d_box, cudaStream_t stream) {

    const int tpb = default_threads_per_block;
    const int B = ceil_divide(K_, tpb);

    k_coords_to_kv_gather<<<B, tpb, 0, stream>>>(
        K_, d_atom_idxs_, d_coords, d_box, d_bin_to_idx_, d_sort_keys_in_, d_sort_vals_in_);

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
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_u,
    cudaStream_t stream) {

    // (ytz) the nonbonded algorithm proceeds as follows:

    // (done in constructor), construct a hilbert curve mapping each of the HILBERT_GRID_DIM x HILBERT_GRID_DIM x HILBERT_GRID_DIM cells into an index.
    // a. decide if we need to rebuild the neighborlist, if so:
    //     - look up which cell each particle belongs to, and its linear index along the hilbert curve.
    //     - use radix pair sort keyed on the hilbert index with values equal to the atomic index
    //     - resulting sorted values is the permutation array.
    //     - permute coords
    // b. else:
    //     - permute new coords
    // c. permute parameters
    // d. compute the nonbonded interactions using the neighborlist
    // e. inverse permute the forces, du/dps into the original index.
    // f. u is buffered into a per-particle array, and then reduced.

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

    const int tpb = default_threads_per_block;

    if (this->needs_sort()) {
        // Sorting always triggers a neighborlist rebuild
        this->sort(d_x, d_box, stream);
    } else {
        // (ytz) see if we need to rebuild the neighborlist.
        k_check_rebuild_coords_and_box_gather<RealType><<<ceil_divide(K_, tpb), tpb, 0, stream>>>(
            K_, d_atom_idxs_, d_x, d_nblist_x_, d_box, d_nblist_box_, nblist_padding_, d_rebuild_nblist_);
        gpuErrchk(cudaPeekAtLastError());

        // we can optimize this away by doing the check on the GPU directly.
        gpuErrchk(cudaMemcpyAsync(
            p_rebuild_nblist_, d_rebuild_nblist_, 1 * sizeof(*p_rebuild_nblist_), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaEventRecord(nblist_flag_sync_event_, stream));
    }
    // compute new coordinates/params
    k_gather_coords_and_params<<<dim3(ceil_divide(K_, tpb), PARAMS_PER_ATOM, 1), tpb, 0, stream>>>(
        K_, d_sorted_atom_idxs_, d_x, d_p, d_gathered_x_, d_gathered_p_);
    gpuErrchk(cudaPeekAtLastError());

    // reset buffers and sorted accumulators
    if (d_du_dx) {
        gpuErrchk(cudaMemsetAsync(d_gathered_du_dx_, 0, K_ * 3 * sizeof(*d_gathered_du_dx_), stream))
    }
    if (d_du_dp) {
        gpuErrchk(cudaMemsetAsync(d_gathered_du_dp_, 0, K_ * PARAMS_PER_ATOM * sizeof(*d_gathered_du_dp_), stream))
    }
    // Syncing to an event allows having additional kernels run while we synchronize
    // Note that if no event is recorded, this is effectively a no-op, such as in the case of sorting.
    gpuErrchk(cudaEventSynchronize(nblist_flag_sync_event_));
    if (p_rebuild_nblist_[0] > 0) {

        nblist_.build_nblist_device(K_, d_gathered_x_, d_box, cutoff_ + nblist_padding_, stream);
        gpuErrchk(cudaMemcpyAsync(
            p_ixn_count_, nblist_.get_ixn_count(), 1 * sizeof(*p_ixn_count_), cudaMemcpyDeviceToHost, stream));

        gpuErrchk(cudaMemcpyAsync(p_box_, d_box, 3 * 3 * sizeof(*d_box), cudaMemcpyDeviceToHost, stream));

        // this stream needs to be synchronized so we can be sure that p_ixn_count_ is properly set.
        gpuErrchk(cudaStreamSynchronize(stream));

        // If there are no interactions, things have broken
        if (p_ixn_count_[0] < 1) {
            throw std::runtime_error("no nonbonded interactions, check system");
        }

        // Verify that the cutoff and box size are valid together. If cutoff is greater than half the box
        // then a particle can interact with multiple periodic copies.
        const double db_cutoff = (cutoff_ + nblist_padding_) * 2;

        // Verify the width of the box in all dimensions is greater than twice the cutoff
        for (int i = 0; i < 3; i++) {
            if (p_box_[i * 3 + i] < db_cutoff) {
                throw std::runtime_error(
                    "Cutoff with padding is more than half of the box width, neighborlist is no longer reliable");
            }
        }

        gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 0, sizeof(*d_rebuild_nblist_), stream));
        gpuErrchk(cudaMemcpyAsync(d_nblist_x_, d_x, N * 3 * sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_nblist_box_, d_box, 3 * 3 * sizeof(*d_box), cudaMemcpyDeviceToDevice, stream));
    }

    // look up which kernel we need for this computation
    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<ceil_divide(p_ixn_count_[0], tpb / warp_size), tpb, 0, stream>>>(
        K_,
        nblist_.get_num_row_idxs(),
        nblist_.get_ixn_count(),
        d_gathered_x_,
        d_gathered_p_,
        d_box,
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

        k_add_ull_to_ull<<<dim3(ceil_divide(N_, tpb), PARAMS_PER_ATOM, 1), tpb, 0, stream>>>(
            N, d_du_dp_buffer_, d_du_dp);
        gpuErrchk(cudaPeekAtLastError());
    }
    // Increment steps
    steps_since_last_sort_++;
}

template <typename RealType>
void NonbondedAllPairs<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    for (int i = 0; i < N; i++) {
        const int idx = i * PARAMS_PER_ATOM;
        const int idx_charge = idx + PARAM_OFFSET_CHARGE;
        const int idx_sig = idx + PARAM_OFFSET_SIG;
        const int idx_eps = idx + PARAM_OFFSET_EPS;
        const int idx_w = idx + PARAM_OFFSET_W;

        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
        du_dp_float[idx_w] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DW>(du_dp[idx_w]);
    }
}

template class NonbondedAllPairs<double>;
template class NonbondedAllPairs<float>;

} // namespace timemachine
