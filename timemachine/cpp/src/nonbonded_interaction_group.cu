#include <complex>
#include <string>
#include <vector>

#include "device_buffer.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "kernels/k_indices.cuh"
#include "nonbonded_common.hpp"
#include "nonbonded_interaction_group.hpp"
#include "set_utils.hpp"
#include <cub/cub.cuh>

#include "k_nonbonded.cuh"

static const int STEPS_PER_SORT = 200;

namespace timemachine {

template <typename RealType>
NonbondedInteractionGroup<RealType>::NonbondedInteractionGroup(
    const int N,
    const std::vector<int> &row_atom_idxs,
    const std::vector<int> &col_atom_idxs,
    const double beta,
    const double cutoff,
    const bool disable_hilbert_sort,
    const double nblist_padding)
    : N_(N), NR_(row_atom_idxs.size()), NC_(col_atom_idxs.size()), sum_storage_bytes_(0),

      kernel_ptrs_({// enumerate over every possible kernel combination
                    // Set threads to 1 if not computing energy to reduced unused shared memory
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP
                    //                                                                 U  X  P
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, 0, 0>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, 0, 1>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, 1, 0>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, 1, 1>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1, 0, 0>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1, 0, 1>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1, 1, 0>,
                    &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1, 1, 1>}),

      beta_(beta), cutoff_(cutoff), steps_since_last_sort_(0), nblist_(N_), nblist_padding_(nblist_padding),
      hilbert_sort_(nullptr), disable_hilbert_(disable_hilbert_sort) {

    this->validate_idxs(N_, row_atom_idxs, col_atom_idxs, false);

    cudaSafeMalloc(&d_col_atom_idxs_, N_ * sizeof(*d_col_atom_idxs_));
    cudaSafeMalloc(&d_row_atom_idxs_, N_ * sizeof(*d_row_atom_idxs_));
    cudaSafeMalloc(&d_u_buffer_, NONBONDED_KERNEL_BLOCKS * sizeof(*d_u_buffer_));

    cudaSafeMalloc(&d_perm_, N_ * sizeof(*d_perm_));

    cudaSafeMalloc(&d_sorted_x_, N_ * 3 * sizeof(*d_sorted_x_));

    cudaSafeMalloc(&d_sorted_p_, N_ * PARAMS_PER_ATOM * sizeof(*d_sorted_p_));
    cudaSafeMalloc(&d_sorted_du_dx_, N_ * 3 * sizeof(*d_sorted_du_dx_));
    cudaSafeMalloc(&d_sorted_du_dp_, N_ * PARAMS_PER_ATOM * sizeof(*d_sorted_du_dp_));

    cudaSafeMalloc(&d_nblist_x_, N_ * 3 * sizeof(*d_nblist_x_));
    gpuErrchk(cudaMemset(d_nblist_x_, 0, N_ * 3 * sizeof(*d_nblist_x_))); // set non-sensical positions
    cudaSafeMalloc(&d_nblist_box_, 3 * 3 * sizeof(*d_nblist_box_));
    gpuErrchk(cudaMemset(d_nblist_box_, 0, 3 * 3 * sizeof(*d_nblist_box_)));
    cudaSafeMalloc(&d_rebuild_nblist_, 1 * sizeof(*d_rebuild_nblist_));
    gpuErrchk(cudaMallocHost(&p_rebuild_nblist_, 1 * sizeof(*p_rebuild_nblist_)));

    gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_, d_u_buffer_, NONBONDED_KERNEL_BLOCKS));

    gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));

    if (!disable_hilbert_) {
        this->hilbert_sort_.reset(new HilbertSort(N_));
    }

    this->set_atom_idxs(row_atom_idxs, col_atom_idxs);

    // Create event with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&nblist_flag_sync_event_, cudaEventDisableTiming));
};

template <typename RealType> NonbondedInteractionGroup<RealType>::~NonbondedInteractionGroup() {
    gpuErrchk(cudaFree(d_col_atom_idxs_));
    gpuErrchk(cudaFree(d_row_atom_idxs_));

    gpuErrchk(cudaFree(d_perm_));

    gpuErrchk(cudaFree(d_sorted_x_));
    gpuErrchk(cudaFree(d_u_buffer_));

    gpuErrchk(cudaFree(d_sorted_p_));
    gpuErrchk(cudaFree(d_sorted_du_dx_));
    gpuErrchk(cudaFree(d_sorted_du_dp_));

    gpuErrchk(cudaFree(d_nblist_x_));
    gpuErrchk(cudaFree(d_nblist_box_));
    gpuErrchk(cudaFree(d_rebuild_nblist_));
    gpuErrchk(cudaFreeHost(p_rebuild_nblist_));

    gpuErrchk(cudaEventDestroy(nblist_flag_sync_event_));

    gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType> bool NonbondedInteractionGroup<RealType>::needs_sort() {
    return steps_since_last_sort_ % STEPS_PER_SORT == 0;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::sort(const double *d_coords, const double *d_box, cudaStream_t stream) {
    // We must rebuild the neighborlist after sorting, as the neighborlist is tied to a particular sort order
    if (!disable_hilbert_) {
        this->hilbert_sort_->sort_device(NR_, d_row_atom_idxs_, d_coords, d_box, d_perm_, stream);
        this->hilbert_sort_->sort_device(NC_, d_col_atom_idxs_, d_coords, d_box, d_perm_ + NR_, stream);
    } else {
        gpuErrchk(cudaMemcpyAsync(
            d_perm_, d_row_atom_idxs_, NR_ * sizeof(*d_row_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(
            d_perm_ + NR_, d_col_atom_idxs_, NC_ * sizeof(*d_col_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
    }
    gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 1, sizeof(*d_rebuild_nblist_), stream));
    // Set the pinned memory to indicate that we need to rebuild
    p_rebuild_nblist_[0] = 1;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,   // N * PARAMS_PER_ATOM
    const double *d_box, // 3 * 3
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
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
            "NonbondedInteractionGroup::execute_device(): expected N == N_, got N=" + std::to_string(N) +
            ", N_=" + std::to_string(N_));
    }

    if (P != N_ * PARAMS_PER_ATOM) {
        throw std::runtime_error(
            "NonbondedInteractionGroup::execute_device(): expected P == N_*" + std::to_string(PARAMS_PER_ATOM) +
            ", got P=" + std::to_string(P) + ", N_*" + std::to_string(PARAMS_PER_ATOM) + "=" +
            std::to_string(N_ * PARAMS_PER_ATOM));
    }

    // If the size of the row or cols is none, exit
    if (NR_ == 0 || NC_ == 0) {
        return;
    }

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int B = ceil_divide(N_, tpb);
    const int K = NR_ + NC_; // total number of interactions
    const int B_K = ceil_divide(K, tpb);

    if (this->needs_sort()) {
        // Sorting always triggers a neighborlist rebuild
        this->sort(d_x, d_box, stream);
    } else {
        // (ytz) see if we need to rebuild the neighborlist.
        // Reuse the d_perm_ here to avoid having to make two kernels calls.
        k_check_rebuild_coords_and_box_gather<RealType><<<B_K, tpb, 0, stream>>>(
            NR_ + NC_, d_perm_, d_x, d_nblist_x_, d_box, d_nblist_box_, nblist_padding_, d_rebuild_nblist_);
        gpuErrchk(cudaPeekAtLastError());
        // we can optimize this away by doing the check on the GPU directly.
        gpuErrchk(cudaMemcpyAsync(
            p_rebuild_nblist_, d_rebuild_nblist_, 1 * sizeof(*p_rebuild_nblist_), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaEventRecord(nblist_flag_sync_event_, stream));
    }

    // compute new coordinates/params
    k_gather_coords_and_params<double, 3, PARAMS_PER_ATOM>
        <<<ceil_divide(K, tpb), tpb, 0, stream>>>(K, d_perm_, d_x, d_p, d_sorted_x_, d_sorted_p_);
    gpuErrchk(cudaPeekAtLastError());
    // reset buffers and sorted accumulators
    if (d_du_dx) {
        gpuErrchk(cudaMemsetAsync(d_sorted_du_dx_, 0, K * 3 * sizeof(*d_sorted_du_dx_), stream))
    }
    if (d_du_dp) {
        gpuErrchk(cudaMemsetAsync(d_sorted_du_dp_, 0, K * PARAMS_PER_ATOM * sizeof(*d_sorted_du_dp_), stream))
    }

    // Syncing to an event allows having additional kernels run while we synchronize
    // Note that if no event is recorded, this is effectively a no-op, such as in the case of sorting.
    gpuErrchk(cudaEventSynchronize(nblist_flag_sync_event_));
    if (p_rebuild_nblist_[0] > 0) {

        nblist_.build_nblist_device(K, d_sorted_x_, d_box, cutoff_ + nblist_padding_, stream);

        gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 0, sizeof(*d_rebuild_nblist_), stream));
        gpuErrchk(cudaMemcpyAsync(d_nblist_x_, d_x, N * 3 * sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_nblist_box_, d_box, 3 * 3 * sizeof(*d_box), cudaMemcpyDeviceToDevice, stream));
    }

    // look up which kernel we need for this computation
    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<NONBONDED_KERNEL_BLOCKS, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, stream>>>(
        K,
        nblist_.get_num_row_idxs(),
        nblist_.get_ixn_count(),
        d_sorted_x_,
        d_sorted_p_,
        d_box,
        beta_,
        cutoff_,
        nblist_.get_row_idxs(),
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_sorted_du_dx_,
        d_sorted_du_dp_,
        d_u == nullptr ? nullptr : d_u_buffer_ // switch to nullptr if we don't request energies
    );

    gpuErrchk(cudaPeekAtLastError());

    // coords are N,3
    if (d_du_dx) {
        k_scatter_accum<unsigned long long, 3><<<B_K, tpb, 0, stream>>>(K, d_perm_, d_sorted_du_dx_, d_du_dx);
        gpuErrchk(cudaPeekAtLastError());
    }

    // params are N, PARAMS_PER_ATOM
    // this needs to be an accumulated permute
    if (d_du_dp) {
        k_scatter_accum<unsigned long long, PARAMS_PER_ATOM>
            <<<B_K, tpb, 0, stream>>>(K, d_perm_, d_sorted_du_dp_, d_du_dp);
        gpuErrchk(cudaPeekAtLastError());
    }
    if (d_u) {
        gpuErrchk(cub::DeviceReduce::Sum(
            d_sum_temp_storage_, sum_storage_bytes_, d_u_buffer_, d_u, NONBONDED_KERNEL_BLOCKS, stream));
    }
    // Increment steps
    steps_since_last_sort_++;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_atom_idxs(
    const std::vector<int> &row_atom_idxs, const std::vector<int> &col_atom_idxs) {

    this->validate_idxs(N_, row_atom_idxs, col_atom_idxs, true);

    std::vector<unsigned int> unsigned_row_idxs = std::vector<unsigned int>(row_atom_idxs.begin(), row_atom_idxs.end());
    std::set<unsigned int> unique_row_atom_idxs(unique_idxs(unsigned_row_idxs));

    std::vector<unsigned int> row_atom_idxs_v(set_to_vector(unique_row_atom_idxs));
    std::vector<unsigned int> col_atom_idxs_v(col_atom_idxs.begin(), col_atom_idxs.end());

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    if (row_atom_idxs_v.size() == 0 || row_atom_idxs_v.size() == N_) {
        this->set_atom_idxs_device(col_atom_idxs_v.size(), row_atom_idxs_v.size(), nullptr, nullptr, stream);
    } else {
        DeviceBuffer<unsigned int> d_col(col_atom_idxs_v);
        DeviceBuffer<unsigned int> d_row(row_atom_idxs_v);

        this->set_atom_idxs_device(col_atom_idxs_v.size(), row_atom_idxs_v.size(), d_col.data, d_row.data, stream);
    }
    gpuErrchk(cudaStreamSynchronize(stream));
}

// set_atom_idxs_device is for use when idxs exist on the GPU already and are used as the new idxs to compute the neighborlist on.
template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_atom_idxs_device(
    const int NC,
    const int NR,
    unsigned int *d_in_column_idxs,
    unsigned int *d_in_row_idxs,
    const cudaStream_t stream) {

    if (NC + NR > N_) {
        throw std::runtime_error("number of idxs must be less than or equal to N");
    }
    if (NR > 0 && NC > 0) {
        const size_t tpb = DEFAULT_THREADS_PER_BLOCK;

        // The indices must already be on the GPU and are copied into the potential's buffers.
        gpuErrchk(cudaMemcpyAsync(
            d_col_atom_idxs_, d_in_column_idxs, NC * sizeof(*d_col_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(
            d_row_atom_idxs_, d_in_row_idxs, NR * sizeof(*d_row_atom_idxs_), cudaMemcpyDeviceToDevice, stream));

        // The neighborlist does not use the indices directly, rather it takes a contiguous set of indices and the ixn group
        // potential will resort the correct particles into the corresponding arrays. We can use the leftover spaces in the
        // two d_*_atom_idxs_ arrays to store these nblist indices.
        // NOTE: The leftover column indices will store the row indices and vice versa.
        k_arange<<<ceil_divide(NR, tpb), tpb, 0, stream>>>(NR, d_col_atom_idxs_ + NC);
        gpuErrchk(cudaPeekAtLastError());
        k_arange<<<ceil_divide(NC, tpb), tpb, 0, stream>>>(NC, d_row_atom_idxs_ + NR, NR);
        gpuErrchk(cudaPeekAtLastError());

        // Resize the nblist
        nblist_.resize_device(NC + NR, stream);

        // Force a NBlist rebuild
        gpuErrchk(cudaMemsetAsync(d_rebuild_nblist_, 1, 1 * sizeof(*d_rebuild_nblist_), stream));

        // Offset into the ends of the arrays that now contain the row and column indices for the nblist
        nblist_.set_idxs_device(NC, NR, d_row_atom_idxs_ + NR, d_col_atom_idxs_ + NC, stream);
    }

    // Update the row and column counts
    this->NR_ = NR;
    this->NC_ = NC;
    // Reset the steps so that we do a new sort
    this->steps_since_last_sort_ = 0;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::du_dp_fixed_to_float(
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

template <typename RealType>
void NonbondedInteractionGroup<RealType>::validate_idxs(
    const int N, const std::vector<int> &row_atom_idxs, const std::vector<int> &col_atom_idxs, const bool allow_empty) {

    if (!allow_empty) {
        if (row_atom_idxs.size() == 0) {
            throw std::runtime_error("row_atom_idxs must be nonempty");
        }
        if (col_atom_idxs.size() == 0) {
            throw std::runtime_error("col_atom_idxs must be nonempty");
        }
        if (row_atom_idxs.size() == static_cast<long unsigned int>(N)) {
            throw std::runtime_error("must be less then N(" + std::to_string(N) + ") row indices");
        }
        if (col_atom_idxs.size() == static_cast<long unsigned int>(N)) {
            throw std::runtime_error("must be less then N(" + std::to_string(N) + ") col indices");
        }
    }
    verify_atom_idxs(N, row_atom_idxs, allow_empty);
    verify_atom_idxs(N, col_atom_idxs, allow_empty);

    // row and col idxs must be disjoint
    std::set<int> unique_row_idxs(row_atom_idxs.begin(), row_atom_idxs.end());
    for (int col_atom_idx : col_atom_idxs) {
        if (unique_row_idxs.find(col_atom_idx) != unique_row_idxs.end()) {
            throw std::runtime_error("row and col indices must be disjoint");
        }
    }
}

template class NonbondedInteractionGroup<double>;
template class NonbondedInteractionGroup<float>;

} // namespace timemachine
