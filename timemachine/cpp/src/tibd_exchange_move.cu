#include "tibd_exchange_move.hpp"

#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_exchange.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_nonbonded.cuh"
#include "kernels/k_probability.cuh"
#include "kernels/k_rotations.cuh"
#include "kernels/k_translations.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"
#include <cub/cub.cuh>
#include <math.h>

// The number of threads per block for the setting of the final weight of the moved mol is low
// if using the same number as in the rest of the kernels of DEFAULT_THREADS_PER_BLOCK
#define WEIGHT_THREADS_PER_BLOCK 512
// Currently only support one sample at a time
#define NUM_SAMPLES 1

namespace timemachine {

template <typename RealType>
TIBDExchangeMove<RealType>::TIBDExchangeMove(
    const int N,
    const std::vector<int> ligand_idxs,
    const std::vector<std::vector<int>> &target_mols,
    const std::vector<double> &params,
    const double temperature,
    const double nb_beta,
    const double cutoff,
    const double radius,
    const int seed,
    const int proposals_per_move)
    : BDExchangeMove<RealType>(N, target_mols, params, temperature, nb_beta, cutoff, seed, proposals_per_move),
      radius_(static_cast<RealType>(radius)), inner_volume_(static_cast<RealType>((4.0 / 3.0) * M_PI * pow(radius, 3))),
      d_rand_states_(DEFAULT_THREADS_PER_BLOCK), d_inner_mols_count_(1), d_inner_mols_(this->num_target_mols_),
      d_outer_mols_count_(1), d_outer_mols_(this->num_target_mols_), d_sorted_indices_(this->num_target_mols_),
      d_sort_storage_(0), d_center_(3), d_acceptance_(round_up_even(2)), d_targeting_inner_vol_(1),
      d_ligand_idxs_(ligand_idxs), d_src_weights_(this->num_target_mols_), d_dest_weights_(this->num_target_mols_),
      d_box_volume_(1), p_inner_count_(1), p_targeting_inner_vol_(1) {

    if (radius <= 0.0) {
        throw std::runtime_error("radius must be greater than 0.0");
    }

    // Create event with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&host_copy_event_, cudaEventDisableTiming));

    k_initialize_curand_states<<<1, DEFAULT_THREADS_PER_BLOCK, 0>>>(
        DEFAULT_THREADS_PER_BLOCK, seed, d_rand_states_.data);
    gpuErrchk(cudaPeekAtLastError());

    // estimate size needed to do radix sorting
    // reuse d_sort_keys_in_ rather than constructing a dummy output idxs buffer
    gpuErrchk(cub::DeviceRadixSort::SortKeys(
        nullptr, sort_storage_bytes_, d_inner_mols_.data, d_sorted_indices_.data, this->num_target_mols_));

    d_sort_storage_.realloc(sort_storage_bytes_);
}

template <typename RealType> TIBDExchangeMove<RealType>::~TIBDExchangeMove() {
    gpuErrchk(cudaEventDestroy(host_copy_event_));
}

template <typename RealType>
void TIBDExchangeMove<RealType>::move_device(
    const int N,
    double *d_coords, // [N, 3]
    double *d_box,    // [3, 3]
    cudaStream_t stream) {

    if (N != this->N_) {
        throw std::runtime_error("N != N_");
    }

    // Set the stream for the generator
    curandErrchk(curandSetStream(this->cr_rng_, stream));

    this->compute_initial_weights(N, d_coords, d_box, stream);

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int mol_blocks = ceil_divide(this->num_target_mols_, tpb);

    dim3 atom_by_atom_grid(ceil_divide(N, tpb), this->mol_size_, 1);

    k_compute_centroid_of_atoms<RealType>
        <<<1, tpb, 0, stream>>>(static_cast<int>(d_ligand_idxs_.length), d_ligand_idxs_.data, d_coords, d_center_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_compute_box_volume<<<1, 1, 0, stream>>>(d_box, d_box_volume_.data);
    gpuErrchk(cudaPeekAtLastError());

    const int num_samples = NUM_SAMPLES;
    for (int move = 0; move < this->proposals_per_move_; move++) {
        // Run only after the first pass, to maintain meaningful `log_probability_host` values
        if (move > 0) {
            // Run a separate kernel to replace the before log probs and weights with the after if accepted a move
            // Need the weights to sample a value and the log probs are just because they aren't expensive to copy
            k_store_accepted_log_probability_targeted<RealType><<<1, tpb, 0>>>(
                this->num_target_mols_,
                d_targeting_inner_vol_.data,
                d_box_volume_.data,
                inner_volume_,
                this->d_acceptance_.data + 1, // Offset to get the last value for the acceptance criteria
                this->d_log_sum_exp_before_.data,
                this->d_log_sum_exp_after_.data,
                this->d_log_weights_before_.data,
                this->d_log_weights_after_.data);
            gpuErrchk(cudaPeekAtLastError());
        }
        // Copy the before log weights to the after weights, we will adjust the after weights incrementally
        gpuErrchk(cudaMemcpyAsync(
            this->d_log_weights_after_.data,
            this->d_log_weights_before_.data,
            this->d_log_weights_after_.size(),
            cudaMemcpyDeviceToDevice,
            stream));

        gpuErrchk(cudaMemsetAsync(d_inner_mols_count_.data, 0, d_inner_mols_count_.size(), stream));
        gpuErrchk(cudaMemsetAsync(d_outer_mols_count_.data, 0, d_outer_mols_count_.size(), stream));

        k_split_mols_inner_outer<RealType><<<mol_blocks, tpb, 0, stream>>>(
            this->num_target_mols_,
            this->d_atom_idxs_.data,
            this->d_mol_offsets_.data,
            d_center_.data,
            radius_ * radius_,
            d_coords,
            d_box,
            d_inner_mols_count_.data,
            d_inner_mols_.data,
            d_outer_mols_count_.data,
            d_outer_mols_.data);
        gpuErrchk(cudaPeekAtLastError());

        // The this->d_acceptance_ buffer contains the random value for determining where to insert and whether to accept the move
        curandErrchk(templateCurandUniform(this->cr_rng_, this->d_acceptance_.data, this->d_acceptance_.length));

        k_decide_targeted_move<<<1, 1, 0, stream>>>(
            this->d_acceptance_.data, d_inner_mols_count_.data, d_outer_mols_count_.data, d_targeting_inner_vol_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Copy count and flag to the host, needed to know how many values to look at for
        // sampling and logsumexp
        gpuErrchk(cudaMemcpyAsync(
            p_inner_count_.data, d_inner_mols_count_.data, d_inner_mols_count_.size(), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(
            p_targeting_inner_vol_.data,
            d_targeting_inner_vol_.data,
            d_targeting_inner_vol_.size(),
            cudaMemcpyDeviceToHost,
            stream));
        gpuErrchk(cudaEventRecord(host_copy_event_, stream));

        k_generate_translations_within_or_outside_a_sphere<<<ceil_divide(num_samples, tpb), tpb, 0, stream>>>(
            num_samples,
            d_box,
            d_center_.data,
            d_targeting_inner_vol_.data,
            radius_,
            d_rand_states_.data,
            this->d_translations_.data);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaEventSynchronize(host_copy_event_));
        int inner_count = p_inner_count_.data[0];

        // Sort the inner mol idxs to ensure deterministic results
        gpuErrchk(cub::DeviceRadixSort::SortKeys(
            d_sort_storage_.data,
            sort_storage_bytes_,
            d_inner_mols_.data,
            d_sorted_indices_.data,
            inner_count,
            0,
            sizeof(*d_sorted_indices_.data) * 8,
            stream));
        gpuErrchk(cudaMemcpyAsync(
            d_sorted_indices_.data,
            d_inner_mols_.data,
            inner_count * sizeof(*d_sorted_indices_.data),
            cudaMemcpyDeviceToHost,
            stream));

        // Sort the outer mol idxs to ensure deterministic results
        gpuErrchk(cub::DeviceRadixSort::SortKeys(
            d_sort_storage_.data,
            sort_storage_bytes_,
            d_outer_mols_.data,
            d_sorted_indices_.data,
            (this->num_target_mols_ - inner_count),
            0,
            sizeof(*d_sorted_indices_.data) * 8,
            stream));

        k_separate_weights_for_targeted<RealType><<<mol_blocks, tpb, 0, stream>>>(
            this->num_target_mols_,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_outer_mols_count_.data,
            d_inner_mols_.data,
            // Avoid an additional copy and directly copy from the sorted array
            d_sorted_indices_.data,
            this->d_log_weights_before_.data,
            d_src_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        // targeting_inner_vol == 1 indicates that we are target the inner volume, starting from the outer mols
        int targeting_inner_vol = p_targeting_inner_vol_.data[0];
        int src_count = targeting_inner_vol == 0 ? inner_count : this->num_target_mols_ - inner_count;
        int dest_count = this->num_target_mols_ - src_count;

        this->logsumexp_.sum_device(src_count, d_src_weights_.data, this->d_log_sum_exp_before_.data, stream);

        this->sampler_.sample_device(src_count, num_samples, d_src_weights_.data, this->d_samples_.data, stream);

        // Selected an index from the src weights, need to remap the samples idx to the mol indices
        k_adjust_sample_idxs<<<ceil_divide(num_samples, tpb), tpb, 0, stream>>>(
            num_samples, targeting_inner_vol == 1 ? d_outer_mols_.data : d_inner_mols_.data, this->d_samples_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Don't move translations into computation of the incremental, as different translations can be used
        // by different bias deletion movers (such as targeted insertion)
        // Don't scale the translations as they computed to be within the region
        this->compute_incremental_weights(N, num_samples, false, d_coords, d_box, stream);

        k_setup_destination_weights_for_targeted<RealType><<<mol_blocks, tpb, 0, stream>>>(
            this->num_target_mols_,
            num_samples,
            this->d_samples_.data,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_outer_mols_count_.data,
            d_inner_mols_.data,
            d_outer_mols_.data,
            this->d_log_weights_after_.data,
            d_dest_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Add one to the destination count, as we just moved a mol there
        this->logsumexp_.sum_device(dest_count + 1, d_dest_weights_.data, this->d_log_sum_exp_after_.data, stream);

        k_attempt_exchange_move_targeted<RealType><<<ceil_divide(N, tpb), tpb, 0, stream>>>(
            N,
            d_targeting_inner_vol_.data,
            d_box_volume_.data,
            inner_volume_,
            this->d_acceptance_.data + 1, // Offset to get the last value for the acceptance criteria
            this->d_log_sum_exp_before_.data,
            this->d_log_sum_exp_after_.data,
            this->d_intermediate_coords_.data,
            d_coords,
            this->d_num_accepted_.data);
        gpuErrchk(cudaPeekAtLastError());
        this->num_attempted_++;
    }
}

template <typename RealType>
std::array<std::vector<double>, 2>
TIBDExchangeMove<RealType>::move_host(const int N, const double *h_coords, const double *h_box) {

    const double box_vol = h_box[0 * 3 + 0] * h_box[1 * 3 + 1] * h_box[2 * 3 + 2];
    if (box_vol <= inner_volume_) {
        throw std::runtime_error("volume of inner radius greater than box volume");
    }

    DeviceBuffer<double> d_coords(N * 3);
    d_coords.copy_from(h_coords);

    DeviceBuffer<double> d_box(3 * 3);
    d_box.copy_from(h_box);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    this->move_device(N, d_coords.data, d_box.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<double> out_coords(d_coords.length);
    d_coords.copy_to(&out_coords[0]);

    std::vector<double> out_box(d_box.length);
    d_box.copy_to(&out_box[0]);

    return std::array<std::vector<double>, 2>({out_coords, out_box});
}

template <typename RealType> double TIBDExchangeMove<RealType>::log_probability_host() {
    std::vector<RealType> h_log_exp_before(2);
    std::vector<RealType> h_log_exp_after(2);
    this->d_log_sum_exp_before_.copy_to(&h_log_exp_before[0]);
    this->d_log_sum_exp_after_.copy_to(&h_log_exp_after[0]);

    int h_targeting_inner_vol;
    d_targeting_inner_vol_.copy_to(&h_targeting_inner_vol);

    RealType h_box_vol;
    d_box_volume_.copy_to(&h_box_vol);

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_before[0]));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_after[0]));

    RealType outer_vol = h_box_vol - inner_volume_;

    RealType log_vol_prob = h_targeting_inner_vol == 1 ? log(inner_volume_) - log(h_box_vol - inner_volume_)
                                                       : log(h_box_vol - inner_volume_) - log(inner_volume_);

    double log_prob = min(static_cast<double>(before_log_prob - after_log_prob + log_vol_prob), 0.0);
    return log_prob;
}

template class TIBDExchangeMove<float>;
template class TIBDExchangeMove<double>;

} // namespace timemachine
