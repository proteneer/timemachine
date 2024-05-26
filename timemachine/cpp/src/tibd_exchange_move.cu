#include "tibd_exchange_move.hpp"

#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_exchange.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_nonbonded.cuh"
#include "kernels/k_probability.cuh"
#include "kernels/k_rotations.cuh"
#include "kernels/k_sampling.cuh"
#include "kernels/k_translations.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"
#include <cub/cub.cuh>
#include <math.h>

namespace timemachine {

// Each step will have 6 values for a translation, first 3 is the inner translation and second 3 is outer translation
static const int TIBD_TRANSLATIONS_PER_STEP_XYZXYZ = 6;

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
    const int num_proposals_per_move,
    const int interval,
    const int batch_size)
    : BDExchangeMove<RealType>(
          N,
          target_mols,
          params,
          temperature,
          nb_beta,
          cutoff,
          seed,
          num_proposals_per_move,
          interval,
          batch_size,
          TIBD_TRANSLATIONS_PER_STEP_XYZXYZ * num_proposals_per_move),
      radius_(static_cast<RealType>(radius)), inner_volume_(static_cast<RealType>((4.0 / 3.0) * M_PI * pow(radius, 3))),
      d_rand_states_(DEFAULT_THREADS_PER_BLOCK), d_inner_mols_count_(1), d_identify_indices_(this->num_target_mols_),
      d_partitioned_indices_(this->num_target_mols_), d_temp_storage_buffer_(0), d_center_(3),
      d_uniform_noise_buffer_(num_proposals_per_move), d_targeting_inner_vol_(this->batch_size_),
      d_ligand_idxs_(ligand_idxs), d_src_log_weights_(this->num_target_mols_ * this->batch_size_),
      d_dest_log_weights_(this->num_target_mols_ * this->batch_size_), d_inner_flags_(this->num_target_mols_),
      d_box_volume_(1),
      d_selected_translations_(this->num_proposals_per_move_ * 3), // TBD: Duplicating some jank... sad
      d_sample_after_segment_offsets_(this->d_sample_segments_offsets_.length),
      d_weights_before_counts_(this->batch_size_), d_weights_after_counts_(this->batch_size_),
      d_lse_max_src_(this->batch_size_), d_lse_exp_sum_src_(this->batch_size_) {

    if (radius <= 0.0) {
        throw std::runtime_error("radius must be greater than 0.0");
    }
    if (d_uniform_noise_buffer_.length != this->d_quaternions_.length / this->QUATERNIONS_PER_STEP) {
        throw std::runtime_error("bug in the code: buffers with random values don't match in batch size");
    }

    // Add 4 to the seed provided to avoid correlating with the four other RNGs
    k_initialize_curand_states<<<
        ceil_divide(d_rand_states_.length, DEFAULT_THREADS_PER_BLOCK),
        DEFAULT_THREADS_PER_BLOCK,
        0>>>(static_cast<int>(d_rand_states_.length), seed + 4, d_rand_states_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_arange<<<ceil_divide(this->num_target_mols_, DEFAULT_THREADS_PER_BLOCK), DEFAULT_THREADS_PER_BLOCK, 0>>>(
        this->num_target_mols_, d_identify_indices_.data, 0);
    gpuErrchk(cudaPeekAtLastError());

    size_t flagged_bytes = 0;
    // Setup buffer for doing the flagged partition
    gpuErrchk(cub::DevicePartition::Flagged(
        nullptr,
        flagged_bytes,
        d_identify_indices_.data,
        d_inner_flags_.data,
        d_partitioned_indices_.data,
        d_inner_mols_count_.data,
        this->num_target_mols_));

    size_t sum_bytes = 0;
    // Will need to compute prefix sums of the count of before and after weights to construct the segment offsets
    gpuErrchk(cub::DeviceScan::InclusiveSum(
        nullptr, sum_bytes, d_weights_before_counts_.data, this->d_sample_segments_offsets_.data, this->batch_size_));
    // Take the larger of the two to use as the temp storage data for CUB
    temp_storage_bytes_ = max(flagged_bytes, sum_bytes);

    // Zero out the sample segments offsets, the first index will always be zero and the inclusive sum will be offset by 1
    gpuErrchk(cudaMemset(this->d_sample_segments_offsets_.data, 0, this->d_sample_segments_offsets_.size()));
    gpuErrchk(cudaMemset(d_sample_after_segment_offsets_.data, 0, d_sample_after_segment_offsets_.size()));
    // Set the inner count to zero and target the inner at the start to ensure that calling `log_probability` produces
    // a zero
    gpuErrchk(cudaMemset(d_inner_mols_count_.data, 0, d_inner_mols_count_.size()));
    std::vector<int> h_targeting_inner(d_targeting_inner_vol_.length, 1);
    d_targeting_inner_vol_.copy_from(&h_targeting_inner[0]);

    // Allocate char as temp_storage_bytes_ is in raw bytes and the type doesn't matter in practice.
    // Equivalent to DeviceBuffer<int> buf(temp_storage_bytes_ / sizeof(int))
    d_temp_storage_buffer_.realloc(temp_storage_bytes_);
}

template <typename RealType> TIBDExchangeMove<RealType>::~TIBDExchangeMove() {}

template <typename RealType>
void TIBDExchangeMove<RealType>::move(
    const int N,
    double *d_coords, // [N, 3]
    double *d_box,    // [3, 3]
    cudaStream_t stream) {

    if (N != this->N_) {
        throw std::runtime_error("N != N_");
    }
    this->step_++;
    if (this->step_ % this->interval_ != 0) {
        return;
    }

    // Set the stream for the generators
    curandErrchk(curandSetStream(this->cr_rng_quat_, stream));
    curandErrchk(curandSetStream(this->cr_rng_translations_, stream));
    curandErrchk(curandSetStream(this->cr_rng_samples_, stream));
    curandErrchk(curandSetStream(this->cr_rng_mh_, stream));

    // Set the offset to 0
    gpuErrchk(cudaMemsetAsync(this->d_noise_offset_.data, 0, this->d_noise_offset_.size(), stream));

    this->compute_initial_log_weights_device(N, d_coords, d_box, stream);

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int mol_blocks = ceil_divide(this->num_target_mols_, tpb);
    const int sample_blocks = ceil_divide(this->batch_size_, tpb);

    dim3 atom_by_atom_grid(ceil_divide(N, tpb), this->mol_size_, 1);

    k_compute_centroid_of_atoms<RealType>
        <<<1, tpb, 0, stream>>>(static_cast<int>(d_ligand_idxs_.length), d_ligand_idxs_.data, d_coords, d_center_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_compute_box_volume<<<1, 1, 0, stream>>>(d_box, d_box_volume_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_flag_mols_inner_outer<RealType><<<mol_blocks, tpb, 0, stream>>>(
        this->num_target_mols_,
        this->d_atom_idxs_.data,
        this->d_mol_offsets_.data,
        d_center_.data,
        radius_ * radius_,
        d_coords,
        d_box,
        d_inner_flags_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Generate all noise upfront for all proposals within a move
    curandErrchk(templateCurandUniform(this->cr_rng_mh_, this->d_mh_noise_.data, this->d_mh_noise_.length));
    // Using the translations RNG from the BDExchangeMove to generate noise for the targeting probability
    curandErrchk(templateCurandUniform(
        this->cr_rng_translations_, this->d_uniform_noise_buffer_.data, this->d_uniform_noise_buffer_.length));
    curandErrchk(
        templateCurandNormal(this->cr_rng_quat_, this->d_quaternions_.data, this->d_quaternions_.length, 0.0, 1.0));
    curandErrchk(
        templateCurandUniform(this->cr_rng_samples_, this->d_sample_noise_.data, this->d_sample_noise_.length));
    k_generate_translations_inside_and_outside_sphere<<<1, d_rand_states_.length, 0, stream>>>(
        this->num_proposals_per_move_, d_box, d_center_.data, radius_, d_rand_states_.data, this->d_translations_.data);
    gpuErrchk(cudaPeekAtLastError());

    /* --Algorithm Description--
    * Targeted Insertion Biased Deletion algorithm is as follows
    *
    * 1. Generate all random noise upfront to ensure bitwise identical results regardless of batch size
    * 2. Compute the initial weights of each of the molecules, called only once per move() (no batching)
    * 3. Copy the initial weights (d_log_weights_before_) to the proposal weight buffers (d_log_weights_after_),
    *    duplicating the values for each proposal in the batch
    * 4. Flag the target molecules that are inside and outside the target region
    * 5. Set up the proposals determining which region is the source region, where waters are to be deleted, and which region is the
    *    destination region, where waters are to be inserted.
    * 6. Construct the offsets for the initial weights (d_src_log_weights_) and after (d_log_weights_after_) log weights
    * 7. Separate out the weights (only evaluate the source region weights) associated with each proposal
    * 8. For each proposal in the batch sample a weight index from the initial weights, aiming to select weights with high energies
    * 9. Compute the logexpsum (using SegmentedSumExp and compute_logsumexp_final) of each proposal's initial weights.
    * 10. Remap the weight indexes selected in 8. back to molecule indexes to be able to determine which coordinates to modify in the next
    *     step.
    * 11. Generate the proposals for all of the sampled molecules in the batch, rotating and translating the mols to the new
    *    positions.
    * 12. For each proposal in the batch separate out the proposals weights (the molecules within the destination region) into the after log weights.
    * 13. Compute the logexpsum (using SegmentedSumExp and compute_logsumexp_final) of each proposal's after log weights.
    * 14. Find the first proposal in the batch that was accepted with the Metropolis-Hastings check
    * 15. If a move was accepted, update the new proposed coordinates and increment the noise offset (d_noise_offset_)
    *     by the index in the batch that was accepted.
    *
    * NOTE: The noise offset is used to determine where in the noise buffers the kernels should look. If a kernel is expecting to
    *       access data beyond the total number of proposals, the kernels leave the buffers untouched. This offset to to
    *       ensure that with a batch size of 1 or 1000 the sequence of proposals is bitwise identical, by using the same noise for
    *       each proposal in the sequence.
    * NOTE: Each proposal has its own initial weights that are the weights of the region where molecules are being deleted from.
    */

    // For the first pass just set the value to zero on the host
    *this->p_noise_offset_.data = 0;
    while (*this->p_noise_offset_.data < this->num_proposals_per_move_) {
        // To ensure determinism between running 1 step per move or K steps per move we have to partition each pass
        // Ordering is consistent, with the tail reversed.
        // https://nvlabs.github.io/cub/structcub_1_1_device_partition.html#a47515ec2a15804719db1b8f3b3124e43
        gpuErrchk(cub::DevicePartition::Flagged(
            d_temp_storage_buffer_.data,
            temp_storage_bytes_,
            d_identify_indices_.data,
            d_inner_flags_.data,
            d_partitioned_indices_.data,
            d_inner_mols_count_.data,
            this->num_target_mols_,
            stream));

        k_decide_targeted_moves<<<sample_blocks, tpb, 0, stream>>>(
            this->num_proposals_per_move_,
            this->batch_size_,
            this->num_target_mols_,
            this->d_noise_offset_.data,
            this->d_uniform_noise_buffer_.data,
            d_inner_mols_count_.data,
            this->d_translations_.data,
            d_targeting_inner_vol_.data,
            d_weights_before_counts_.data,
            d_weights_after_counts_.data,
            d_selected_translations_.data);
        gpuErrchk(cudaPeekAtLastError());

        // TBD: Combine the two inclusive sums into a single large inclusive sum
        gpuErrchk(cub::DeviceScan::InclusiveSum(
            d_temp_storage_buffer_.data,
            temp_storage_bytes_,
            d_weights_before_counts_.data,
            this->d_sample_segments_offsets_.data + 1, // Offset by one as the first idx is always 0
            this->batch_size_,
            stream));

        gpuErrchk(cub::DeviceScan::InclusiveSum(
            d_temp_storage_buffer_.data,
            temp_storage_bytes_,
            d_weights_after_counts_.data,
            d_sample_after_segment_offsets_.data + 1, // Offset by one as the first idx is always 0
            this->batch_size_,
            stream));

        k_separate_weights_for_targeted<RealType><<<dim3(mol_blocks, this->batch_size_, 1), tpb, 0, stream>>>(
            this->batch_size_,
            this->num_target_mols_,
            this->d_sample_segments_offsets_.data,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_partitioned_indices_.data,
            this->d_log_weights_before_.data,
            d_src_log_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Have to construct the gumbel buffer directly here to get the sampler to be bitwise deterministic
        // refer to k_setup_gumbel_max_trick_targeted_insertion impl documentation.
        k_setup_gumbel_max_trick_targeted_insertion<RealType>
            <<<dim3(mol_blocks, this->batch_size_, 1), tpb, 0, stream>>>(
                this->batch_size_,
                this->num_target_mols_,
                static_cast<int>(this->d_sample_noise_.length),
                this->d_noise_offset_.data,
                this->d_sample_segments_offsets_.data,
                d_src_log_weights_.data,
                this->d_sample_noise_.data,
                this->d_sampling_intermediate_.data);
        gpuErrchk(cudaPeekAtLastError());

        this->sampler_.sample_given_gumbel_noise_device(
            this->batch_size_,
            this->d_sample_segments_offsets_.data,
            this->d_sampling_intermediate_.data,
            this->d_samples_.data,
            stream);

        this->logsumexp_.sum_device(
            this->num_target_mols_ * this->batch_size_,
            this->batch_size_,
            this->d_sample_segments_offsets_.data,
            d_src_log_weights_.data,
            d_lse_max_src_.data,
            d_lse_exp_sum_src_.data,
            stream);

        // Selected an index from the src weights, need to remap the samples idx to the mol indices
        k_adjust_sample_idxs<<<sample_blocks, tpb, 0, stream>>>(
            this->batch_size_,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_partitioned_indices_.data,
            this->d_samples_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Don't move translations into computation of the incremental, as different translations can be used
        // by different bias deletion movers (such as targeted insertion)
        // Don't scale the translations as they are computed to be within the targeted region
        this->compute_incremental_log_weights_device(
            N, false, d_box, d_coords, this->d_quaternions_.data, this->d_selected_translations_.data, stream);

        k_setup_destination_weights_for_targeted<RealType><<<dim3(mol_blocks, this->batch_size_, 1), tpb, 0, stream>>>(
            this->batch_size_,
            this->num_target_mols_,
            this->d_samples_.data,
            d_sample_after_segment_offsets_.data,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_partitioned_indices_.data,
            this->d_log_weights_after_.data,
            d_dest_log_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        this->logsumexp_.sum_device(
            this->num_target_mols_ * this->batch_size_,
            this->batch_size_,
            d_sample_after_segment_offsets_.data,
            d_dest_log_weights_.data,
            this->d_lse_max_after_.data,
            this->d_lse_exp_sum_after_.data,
            stream);

        k_accept_first_valid_move_targeted<RealType><<<1, min(512, this->batch_size_), 0, stream>>>(
            this->num_proposals_per_move_,
            this->num_target_mols_,
            this->batch_size_,
            inner_volume_,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_box_volume_.data,
            this->d_noise_offset_.data,
            this->d_samples_.data,
            d_lse_max_src_.data,
            d_lse_exp_sum_src_.data,
            this->d_lse_max_after_.data,
            this->d_lse_exp_sum_after_.data,
            this->d_mh_noise_.data,
            this->d_selected_sample_.data);
        gpuErrchk(cudaPeekAtLastError());

        k_store_exchange_move<<<mol_blocks, tpb, 0, stream>>>(
            this->batch_size_,
            this->num_target_mols_,
            this->d_selected_sample_.data,
            this->d_samples_.data,
            this->d_target_mol_offsets_.data,
            this->d_sample_segments_offsets_.data,
            this->d_intermediate_coords_.data,
            d_coords,
            this->d_before_mol_energy_buffer_.data,
            this->d_proposal_mol_energy_buffer_.data,
            this->d_noise_offset_.data,
            d_inner_flags_.data,
            this->d_num_accepted_.data);
        gpuErrchk(cudaPeekAtLastError());
        k_convert_energies_to_log_weights<RealType><<<mol_blocks, tpb, 0, stream>>>(
            this->num_target_mols_,
            this->beta_,
            this->d_before_mol_energy_buffer_.data,
            this->d_log_weights_before_.data);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpyAsync(
            this->p_noise_offset_.data,
            this->d_noise_offset_.data,
            this->d_noise_offset_.size(),
            cudaMemcpyDeviceToHost,
            stream));
        // Synchronize to get the new offset
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    this->num_attempted_ += this->num_proposals_per_move_;
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

    this->move(N, d_coords.data, d_box.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<double> out_coords(d_coords.length);
    d_coords.copy_to(&out_coords[0]);

    std::vector<double> out_box(d_box.length);
    d_box.copy_to(&out_box[0]);

    return std::array<std::vector<double>, 2>({out_coords, out_box});
}

template <typename RealType> double TIBDExchangeMove<RealType>::raw_log_probability_host() {
    std::vector<RealType> h_log_exp_src(2 * this->batch_size_);
    std::vector<RealType> h_log_exp_after(2 * this->batch_size_);
    d_lse_max_src_.copy_to(&h_log_exp_src[0]);
    d_lse_exp_sum_src_.copy_to(&h_log_exp_src[this->batch_size_]);
    this->d_lse_max_after_.copy_to(&h_log_exp_after[0]);
    this->d_lse_exp_sum_after_.copy_to(&h_log_exp_after[this->batch_size_]);

    int h_targeting_inner_vol[this->batch_size_];
    d_targeting_inner_vol_.copy_to(h_targeting_inner_vol);

    int h_local_inner_count[1];
    d_inner_mols_count_.copy_to(h_local_inner_count);

    RealType h_box_vol;
    d_box_volume_.copy_to(&h_box_vol);

    RealType outer_vol = h_box_vol - inner_volume_;

    const RealType raw_log_acceptance = compute_raw_log_probability_targeted<RealType>(
        h_targeting_inner_vol[0],
        inner_volume_,
        outer_vol,
        h_local_inner_count[0],
        this->num_target_mols_,
        &h_log_exp_src[0],
        &h_log_exp_src[this->batch_size_],
        &h_log_exp_after[0],
        &h_log_exp_after[this->batch_size_]);

    return static_cast<double>(raw_log_acceptance);
}

template <typename RealType> double TIBDExchangeMove<RealType>::log_probability_host() {
    return min(raw_log_probability_host(), 0.0);
}

template class TIBDExchangeMove<float>;
template class TIBDExchangeMove<double>;

} // namespace timemachine
