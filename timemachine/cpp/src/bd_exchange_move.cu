#include "bd_exchange_move.hpp"

#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_exchange.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_nonbonded.cuh"
#include "kernels/k_probability.cuh"
#include "kernels/k_rotations.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"

namespace timemachine {
// The number of threads per block for the setting of the final weight of the moved mol is low
// if using the same number as in the rest of the kernels of DEFAULT_THREADS_PER_BLOCK
static const int WEIGHT_THREADS_PER_BLOCK = 512;
// The number of translations to generate each step.
static const int BD_TRANSLATIONS_PER_STEP_XYZ = 3;

template <typename RealType>
BDExchangeMove<RealType>::BDExchangeMove(
    const int N,
    const std::vector<std::vector<int>> &target_mols,
    const std::vector<double> &params,
    const double temperature,
    const double nb_beta,
    const double cutoff,
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
          BD_TRANSLATIONS_PER_STEP_XYZ * num_proposals_per_move) {}

template <typename RealType>
BDExchangeMove<RealType>::BDExchangeMove(
    const int N,
    const std::vector<std::vector<int>> &target_mols,
    const std::vector<double> &params,
    const double temperature,
    const double nb_beta,
    const double cutoff,
    const int seed,
    const int num_proposals_per_move,
    const int interval,
    const int batch_size,
    const int translation_buffer_size)
    : Mover(interval), N_(N), mol_size_(target_mols[0].size()), num_proposals_per_move_(num_proposals_per_move),
      steps_per_move_(num_proposals_per_move_ / batch_size), num_target_mols_(target_mols.size()),
      nb_beta_(static_cast<RealType>(nb_beta)), beta_(static_cast<RealType>(1.0 / (BOLTZ * temperature))),
      cutoff_squared_(static_cast<RealType>(cutoff * cutoff)), batch_size_(batch_size),
      num_intermediates_per_reduce_(ceil_divide(N_, WEIGHT_THREADS_PER_BLOCK)), num_attempted_(0),
      mol_potential_(N, target_mols, nb_beta, cutoff), sampler_(num_target_mols_, batch_size_, seed),
      logsumexp_(num_target_mols_, batch_size_), d_intermediate_coords_(batch_size_ * mol_size_ * 3), d_params_(params),
      d_before_mol_energy_buffer_(num_target_mols_), d_proposal_mol_energy_buffer_(num_target_mols_ * batch_size_),
      d_sample_per_atom_energy_buffer_(batch_size_ * mol_size_ * N), d_atom_idxs_(get_atom_indices(target_mols)),
      d_mol_offsets_(get_mol_offsets(target_mols)), d_log_weights_before_(num_target_mols_),
      d_log_weights_after_(batch_size_ * num_target_mols_), d_lse_max_before_(1), d_lse_exp_sum_before_(1),
      d_lse_max_after_(batch_size_), d_lse_exp_sum_after_(batch_size_), d_samples_(batch_size_), d_selected_sample_(1),
      d_quaternions_(round_up_even(QUATERNIONS_PER_STEP * num_proposals_per_move_)),
      d_mh_noise_(num_proposals_per_move), d_num_accepted_(1), d_target_mol_atoms_(batch_size_ * mol_size_),
      d_target_mol_offsets_(num_target_mols_ + 1),
      d_intermediate_sample_weights_(batch_size_ * num_intermediates_per_reduce_),
      d_sample_noise_(num_target_mols_ * num_proposals_per_move_),
      d_sampling_intermediate_(num_target_mols_ * batch_size_), d_translations_(translation_buffer_size),
      d_sample_segments_offsets_(batch_size_ + 1), d_noise_offset_(1), p_noise_offset_(1) {

    if (num_proposals_per_move_ <= 0) {
        throw std::runtime_error("proposals per move must be greater than 0");
    }
    if (mol_size_ == 0) {
        throw std::runtime_error("must provide non-empty molecule indices");
    }
    verify_mols_contiguous(target_mols);
    for (int i = 0; i < target_mols.size(); i++) {
        if (target_mols[i].size() != mol_size_) {
            throw std::runtime_error("only support running with mols with constant size, got mixed sizes");
        }
    }
    // Clear out the logsumexp values so the log probability starts off as zero
    gpuErrchk(cudaMemset(d_lse_exp_sum_before_.data, 0, d_lse_exp_sum_before_.size()));
    gpuErrchk(cudaMemset(d_lse_max_before_.data, 0, d_lse_max_before_.size()));
    gpuErrchk(cudaMemset(d_lse_exp_sum_after_.data, 0, d_lse_exp_sum_after_.size()));
    gpuErrchk(cudaMemset(d_lse_max_after_.data, 0, d_lse_max_after_.size()));
    gpuErrchk(cudaMemset(d_num_accepted_.data, 0, d_num_accepted_.size()));

    // Initialize several different RNGs to allow for determinism between numbers of steps per move
    curandErrchk(curandCreateGenerator(&cr_rng_quat_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_quat_, seed));

    curandErrchk(curandCreateGenerator(&cr_rng_translations_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_translations_, seed + 1));

    curandErrchk(curandCreateGenerator(&cr_rng_samples_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_samples_, seed + 2));

    curandErrchk(curandCreateGenerator(&cr_rng_mh_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_mh_, seed + 3));

    // Setup the sample segments
    // constant for BDExchangeMove since the sample size is always batches of num_target_mols_ weights
    std::vector<int> h_sample_segments(d_sample_segments_offsets_.length);
    int offset = 0;
    for (unsigned int i = 0; i < h_sample_segments.size(); i++) {
        h_sample_segments[i] = offset;
        offset += num_target_mols_;
    }
    d_sample_segments_offsets_.copy_from(&h_sample_segments[0]);
}

template <typename RealType> BDExchangeMove<RealType>::~BDExchangeMove() {
    curandErrchk(curandDestroyGenerator(cr_rng_quat_));
    curandErrchk(curandDestroyGenerator(cr_rng_translations_));
    curandErrchk(curandDestroyGenerator(cr_rng_samples_));
    curandErrchk(curandDestroyGenerator(cr_rng_mh_));
}

template <typename RealType>
void BDExchangeMove<RealType>::move(
    const int N,
    double *d_coords, // [N, 3]
    double *d_box,    // [3, 3]
    cudaStream_t stream) {

    if (N != N_) {
        throw std::runtime_error("N != N_");
    }
    this->step_++;
    if (this->step_ % this->interval_ != 0) {
        return;
    }
    if (d_translations_.length / BD_TRANSLATIONS_PER_STEP_XYZ !=
        this->d_quaternions_.length / this->QUATERNIONS_PER_STEP) {
        throw std::runtime_error("bug in the code: buffers with random values don't match in batch size");
    }

    // Set the stream for the generators
    curandErrchk(curandSetStream(cr_rng_quat_, stream));
    curandErrchk(curandSetStream(cr_rng_translations_, stream));
    curandErrchk(curandSetStream(cr_rng_samples_, stream));
    curandErrchk(curandSetStream(cr_rng_mh_, stream));

    // Set the offset to 0
    gpuErrchk(cudaMemsetAsync(d_noise_offset_.data, 0, d_noise_offset_.size(), stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    /* --Algorithm Description--
    * Biased Deletion is done in several steps
    * 1. Generate all random noise upfront to ensure bitwise identical results regardless of batch size
    * 2. Compute the initial weights of each of the molecules (no batching)
    * 3. Copy the initial weights (d_log_weights_before_) to the proposal weight buffers (d_log_weights_after_),
    *    duplicating the values for each proposal in the batch
    * 4. For each proposal in the batch sample a molecule from the initial weights, aiming to select molecules with high energies
    * 5. Generate the proposals for all of the sampled molecules in the batch, rotating and translating the mols to the new
    *    positions.
    * 6. Compute the weights for each of the proposals in the batch
    * 7. Compute the logexpsum (using SegmentedSumExp and compute_logsumexp_final) of each set of proposal weights
    * 8. Find the first proposal in the batch that was accepted with the Metropolis-Hastings check
    * 9. If a move was accepted, update the new proposed coordinates and increment the noise offset (d_noise_offset_)
    *    by the value in the batch that was accepted.
    * 10. If running another move, copy the accepted weights, if any, to the initial weights buffer. Return to 4
    *
    * NOTE: The noise offset is used to determine where in the noise buffers the kernels should look. If a kernel is expecting to
    *       access data beyond the total number of proposals, the kernels leave the buffers untouched. This offset to to
    *       ensure that with a batch size of 1 or 1000 the sequence of proposals is bitwise identical, by using the same noise for
    *       each proposal in the sequence.
    */

    this->compute_initial_log_weights_device(N, d_coords, d_box, stream);

    // Compute logsumexp of energies once upfront to get log probabilities
    logsumexp_.sum_device(
        num_target_mols_,
        1,
        d_sample_segments_offsets_.data,
        d_log_weights_before_.data,
        d_lse_max_before_.data,
        d_lse_exp_sum_before_.data,
        stream);

    // All of the noise is generated upfront
    curandErrchk(templateCurandNormal(cr_rng_quat_, d_quaternions_.data, d_quaternions_.length, 0.0, 1.0));
    curandErrchk(templateCurandUniform(cr_rng_translations_, d_translations_.data, d_translations_.length));
    curandErrchk(templateCurandUniform(cr_rng_samples_, d_sample_noise_.data, d_sample_noise_.length));
    curandErrchk(templateCurandUniform(cr_rng_mh_, d_mh_noise_.data, d_mh_noise_.length));
    // For the first pass just set the value to zero on the host
    *p_noise_offset_.data = 0;
    while (*p_noise_offset_.data < num_proposals_per_move_) {
        if (*p_noise_offset_.data > 0) {
            // Run only after the first pass, to maintain meaningful `log_probability_host` values
            // Run a separate kernel to replace the before logsumexp values with the after if accepted a move
            // Could also recompute the logsumexp each round, but more expensive than probably necessary.
            k_store_accepted_log_probability<RealType><<<1, 1, 0>>>(
                num_target_mols_,
                batch_size_,
                d_selected_sample_.data,
                d_lse_max_before_.data,
                d_lse_exp_sum_before_.data,
                d_lse_max_after_.data,
                d_lse_exp_sum_after_.data);
            gpuErrchk(cudaPeekAtLastError());
        }

        sampler_.sample_given_noise_and_offset_device(
            num_target_mols_ * batch_size_,
            batch_size_,
            num_proposals_per_move_,
            d_sample_segments_offsets_.data,
            d_log_weights_before_.data,
            d_noise_offset_.data,
            d_sample_noise_.data,
            d_sampling_intermediate_.data,
            d_samples_.data,
            stream);

        // Don't move translations into computation of the incremental, as different translations can be used
        // by different bias deletion movers (such as targeted insertion)
        // scale the translations as they are between [0, 1]
        this->compute_incremental_log_weights_device(
            N, true, d_box, d_coords, this->d_quaternions_.data, this->d_translations_.data, stream);

        logsumexp_.sum_device(
            num_target_mols_ * batch_size_,
            batch_size_,
            d_sample_segments_offsets_.data,
            d_log_weights_after_.data,
            d_lse_max_after_.data,
            d_lse_exp_sum_after_.data,
            stream);

        k_accept_first_valid_move<RealType><<<1, min(512, batch_size_), 0, stream>>>(
            num_proposals_per_move_,
            num_target_mols_,
            batch_size_,
            d_noise_offset_.data,
            d_samples_.data,
            d_lse_max_before_.data,
            d_lse_exp_sum_before_.data,
            d_lse_max_after_.data,
            d_lse_exp_sum_after_.data,
            d_mh_noise_.data,
            d_selected_sample_.data);
        gpuErrchk(cudaPeekAtLastError());

        k_store_exchange_move<<<ceil_divide(num_target_mols_, tpb), tpb, 0, stream>>>(
            batch_size_,
            num_target_mols_,
            d_selected_sample_.data,
            d_samples_.data,
            d_target_mol_offsets_.data,
            d_sample_segments_offsets_.data,
            d_intermediate_coords_.data,
            d_coords,
            d_before_mol_energy_buffer_.data,
            d_proposal_mol_energy_buffer_.data,
            d_noise_offset_.data,
            nullptr, // No inner/outer flags in Biased deletion
            d_num_accepted_.data);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpyAsync(
            p_noise_offset_.data, d_noise_offset_.data, d_noise_offset_.size(), cudaMemcpyDeviceToHost, stream));
        // Synchronize to get the new offset
        gpuErrchk(cudaStreamSynchronize(stream));
        k_convert_energies_to_log_weights<RealType><<<ceil_divide(num_target_mols_, tpb), tpb, 0, stream>>>(
            num_target_mols_, beta_, d_before_mol_energy_buffer_.data, d_log_weights_before_.data);
        gpuErrchk(cudaPeekAtLastError());
    }
    // Number of attempts is always the number of proposals per moves
    num_attempted_ += num_proposals_per_move_;
}

template <typename RealType>
void BDExchangeMove<RealType>::compute_initial_log_weights_device(
    const int N, double *d_coords, double *d_box, cudaStream_t stream) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int mol_blocks = ceil_divide(num_target_mols_, tpb);
    mol_potential_.mol_energies_device(
        N,
        num_target_mols_,
        d_coords,
        d_params_.data,
        d_box,
        d_before_mol_energy_buffer_.data, // Don't need to zero, will be overridden
        stream);

    k_convert_energies_to_log_weights<RealType><<<mol_blocks, tpb, 0, stream>>>(
        num_target_mols_, beta_, d_before_mol_energy_buffer_.data, d_log_weights_before_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Copy the same mol energies repeatedly from the before energies to the proposal energies
    k_copy_batch<__int128><<<dim3(ceil_divide(num_target_mols_, tpb), batch_size_, 1), tpb, 0, stream>>>(
        num_target_mols_, batch_size_, d_before_mol_energy_buffer_.data, d_proposal_mol_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void BDExchangeMove<RealType>::compute_incremental_log_weights_device(
    const int N,
    const bool scale,
    const double *d_box,            // [3, 3]
    const double *d_coords,         // [N, 3]
    const RealType *d_quaternions,  // [batch_size_, 4]
    const RealType *d_translations, // [batch_size_, 3]
    cudaStream_t stream) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    dim3 atom_by_atom_grid(ceil_divide(N, tpb), mol_size_ * batch_size_, 1);

    k_setup_proposals<<<ceil_divide(batch_size_, tpb), tpb, 0, stream>>>(
        num_proposals_per_move_,
        batch_size_,
        mol_size_,
        d_noise_offset_.data,
        d_samples_.data,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_target_mol_atoms_.data,
        d_target_mol_offsets_.data);
    gpuErrchk(cudaPeekAtLastError());

    if (scale) {
        k_rotate_and_translate_mols<RealType, true><<<ceil_divide(batch_size_, tpb), tpb, 0, stream>>>(
            num_proposals_per_move_,
            batch_size_,
            d_noise_offset_.data,
            d_coords,
            d_box,
            d_samples_.data,
            d_target_mol_offsets_.data,
            d_quaternions,
            d_translations,
            d_intermediate_coords_.data);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        k_rotate_and_translate_mols<RealType, false><<<ceil_divide(batch_size_, tpb), tpb, 0, stream>>>(
            num_proposals_per_move_,
            batch_size_,
            d_noise_offset_.data,
            d_coords,
            d_box,
            d_samples_.data,
            d_target_mol_offsets_.data,
            d_quaternions,
            d_translations,
            d_intermediate_coords_.data);
        gpuErrchk(cudaPeekAtLastError());
    }

    k_atom_by_atom_energies<RealType><<<atom_by_atom_grid, tpb, 0, stream>>>(
        N,
        mol_size_ * batch_size_,
        d_target_mol_atoms_.data,
        nullptr,
        d_coords,
        d_params_.data,
        d_box,
        nb_beta_,
        cutoff_squared_,
        d_sample_per_atom_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Subtract off the weights for the individual waters from the sampled water.
    // It modifies the sampled mol energy value, leaving it in an invalid state, which is why
    // we later call k_set_sampled_energy to set the weight of the sampled mol
    k_adjust_energies<RealType, true><<<dim3(ceil_divide(num_target_mols_, tpb), batch_size_, 1), tpb, 0, stream>>>(
        N,
        batch_size_,
        mol_size_,
        num_target_mols_,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_sample_per_atom_energy_buffer_.data,
        d_proposal_mol_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_atom_by_atom_energies<RealType><<<atom_by_atom_grid, tpb, 0, stream>>>(
        N,
        mol_size_ * batch_size_,
        d_target_mol_atoms_.data,
        d_intermediate_coords_.data,
        d_coords,
        d_params_.data,
        d_box,
        nb_beta_,
        cutoff_squared_,
        d_sample_per_atom_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Add in the new weights from the individual waters
    // the sampled weight continues to be garbage
    k_adjust_energies<RealType, false><<<dim3(ceil_divide(num_target_mols_, tpb), batch_size_, 1), tpb, 0, stream>>>(
        N,
        batch_size_,
        mol_size_,
        num_target_mols_,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_sample_per_atom_energy_buffer_.data,
        d_proposal_mol_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Set the sampled weight to be the correct value
    k_set_sampled_energy_block<RealType, WEIGHT_THREADS_PER_BLOCK>
        <<<dim3(num_intermediates_per_reduce_, batch_size_, 1), WEIGHT_THREADS_PER_BLOCK, 0, stream>>>(
            N,
            batch_size_,
            mol_size_,
            num_target_mols_,
            d_target_mol_atoms_.data,
            d_sample_per_atom_energy_buffer_.data,
            d_intermediate_sample_weights_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_set_sampled_energy_reduce<WEIGHT_THREADS_PER_BLOCK>
        <<<dim3(1, batch_size_, 1), WEIGHT_THREADS_PER_BLOCK, 0, stream>>>(
            batch_size_,
            num_target_mols_,
            num_intermediates_per_reduce_,       // Number of intermediates per sample in batch
            d_samples_.data,                     // where to set the value
            d_intermediate_sample_weights_.data, // intermediate fixed point weights
            d_proposal_mol_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_convert_energies_to_log_weights<RealType><<<ceil_divide(num_target_mols_ * batch_size_, tpb), tpb, 0, stream>>>(
        num_target_mols_ * batch_size_, beta_, d_proposal_mol_energy_buffer_.data, d_log_weights_after_.data);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
std::vector<std::vector<RealType>> BDExchangeMove<RealType>::compute_incremental_log_weights_host(
    const int N,
    const double *h_coords, // [N, 3]
    const double *h_box,    // [3, 3]
    const int *h_mol_idxs,
    const RealType *h_quaternions, // [batch_size_, 4]
    const RealType *h_translations // [batch_size_, 3]
) {
    if (N != N_) {
        throw std::runtime_error("N != N_");
    }

    DeviceBuffer<double> d_coords(N * 3);
    DeviceBuffer<double> d_box(3 * 3);

    d_coords.copy_from(h_coords);
    d_box.copy_from(h_box);

    d_quaternions_.copy_from(h_quaternions);
    d_translations_.copy_from(h_translations);
    d_samples_.copy_from(h_mol_idxs);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    // Set the offset to 0
    gpuErrchk(cudaMemsetAsync(d_noise_offset_.data, 0, d_noise_offset_.size(), stream));

    // Setup the initial weights
    this->compute_initial_log_weights_device(N, d_coords.data, d_box.data, stream);

    this->compute_incremental_log_weights_device(
        N,
        false, // Never scale the translations here, expect the user to do that in python
        d_box.data,
        d_coords.data,
        d_quaternions_.data,
        d_translations_.data,
        stream);

    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<RealType> h_log_weights_after(d_log_weights_after_.length);
    d_log_weights_after_.copy_to(&h_log_weights_after[0]);

    std::vector<int> h_sample_segments_offsets(d_sample_segments_offsets_.length);
    d_sample_segments_offsets_.copy_to(&h_sample_segments_offsets[0]);

    std::vector<std::vector<RealType>> h_output(this->batch_size_);
    for (unsigned int i = 0; i < this->batch_size_; i++) {
        int start = h_sample_segments_offsets[i];
        int end = h_sample_segments_offsets[i + 1];
        h_output[i] = std::vector<RealType>(h_log_weights_after.begin() + start, h_log_weights_after.begin() + end);
    }
    return h_output;
}

template <typename RealType>
std::vector<RealType> BDExchangeMove<RealType>::compute_initial_log_weights_host(
    const int N,
    const double *h_coords, // [N, 3]
    const double *h_box     // [3, 3]
) {
    if (N != N_) {
        throw std::runtime_error("N != N_");
    }

    DeviceBuffer<double> d_coords(N * 3);
    DeviceBuffer<double> d_box(3 * 3);

    d_coords.copy_from(h_coords);
    d_box.copy_from(h_box);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    // Setup the initial weights
    this->compute_initial_log_weights_device(N, d_coords.data, d_box.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    return this->get_before_log_weights();
}

template <typename RealType> std::vector<RealType> BDExchangeMove<RealType>::get_before_log_weights() {
    std::vector<RealType> h_before_log_weights(d_log_weights_before_.length);
    d_log_weights_before_.copy_to(&h_before_log_weights[0]);

    return h_before_log_weights;
}

template <typename RealType> std::vector<RealType> BDExchangeMove<RealType>::get_after_log_weights() {
    std::vector<RealType> h_after_log_weights(d_log_weights_after_.length);
    d_log_weights_after_.copy_to(&h_after_log_weights[0]);

    return h_after_log_weights;
}

template <typename RealType> double BDExchangeMove<RealType>::raw_log_probability_host() {
    std::vector<RealType> h_log_exp_before(2);
    // In the case of batch size > 1 need to increase the amount of data copied
    std::vector<RealType> h_log_exp_after(2 * batch_size_);
    d_lse_max_before_.copy_to(&h_log_exp_before[0]);
    d_lse_exp_sum_before_.copy_to(&h_log_exp_before[1]);
    d_lse_max_after_.copy_to(&h_log_exp_after[0]);
    d_lse_exp_sum_after_.copy_to(&h_log_exp_after[batch_size_]);

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final(h_log_exp_before[0], h_log_exp_before[1]));
    RealType after_log_prob =
        convert_nan_to_inf(compute_logsumexp_final(h_log_exp_after[0], h_log_exp_after[batch_size_]));

    return static_cast<double>(before_log_prob - after_log_prob);
}

template <typename RealType> double BDExchangeMove<RealType>::log_probability_host() {
    return min(raw_log_probability_host(), 0.0);
}

template <typename RealType> size_t BDExchangeMove<RealType>::n_accepted() const {
    size_t h_accepted;
    d_num_accepted_.copy_to(&h_accepted);
    return h_accepted;
}

template <typename RealType> std::vector<double> BDExchangeMove<RealType>::get_params() {
    std::vector<double> h_params(d_params_.length);
    d_params_.copy_to(&h_params[0]);
    return h_params;
};

template <typename RealType> void BDExchangeMove<RealType>::set_params(const std::vector<double> &params) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    DeviceBuffer<double> d_params(params.size());
    d_params.copy_from(&params[0]);
    this->set_params_device(params.size(), d_params.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
};

template <typename RealType>
void BDExchangeMove<RealType>::set_params_device(const int size, const double *d_p, const cudaStream_t stream) {
    if (d_params_.length != size) {
        throw std::runtime_error("number of params don't match");
    }
    gpuErrchk(cudaMemcpyAsync(d_params_.data, d_p, d_params_.size(), cudaMemcpyDeviceToDevice, stream));
};

template class BDExchangeMove<float>;
template class BDExchangeMove<double>;

} // namespace timemachine
