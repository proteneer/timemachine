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
// The number of translations to generate each step. The first three values are a unit vector translation and the fourth
// value is used for the metropolis hasting check
static const int BD_TRANSLATIONS_PER_STEP_XYZW = 4;

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
          round_up_even(BD_TRANSLATIONS_PER_STEP_XYZW * num_proposals_per_move)) {}

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
      cutoff_squared_(static_cast<RealType>(cutoff * cutoff)), batch_size_(batch_size), num_attempted_(0),
      mol_potential_(N, target_mols, nb_beta, cutoff), sampler_(num_target_mols_, batch_size_, seed),
      logsumexp_(num_target_mols_, batch_size_), d_intermediate_coords_(batch_size_ * mol_size_ * 3), d_params_(params),
      d_mol_energy_buffer_(batch_size_ * num_target_mols_),
      d_sample_per_atom_energy_buffer_(batch_size_ * mol_size_ * N), d_atom_idxs_(get_atom_indices(target_mols)),
      d_mol_offsets_(get_mol_offsets(target_mols)), d_log_weights_before_(num_target_mols_),
      d_log_weights_after_(num_target_mols_), d_lse_max_before_(1), d_lse_exp_sum_before_(1),
      d_lse_max_after_(batch_size_), d_lse_exp_sum_after_(batch_size_), d_samples_(batch_size_),
      d_quaternions_(round_up_even(QUATERNIONS_PER_STEP * num_proposals_per_move_ * batch_size_)), d_num_accepted_(1),
      d_target_mol_atoms_(batch_size_ * mol_size_), d_target_mol_offsets_(num_target_mols_ + 1),
      d_intermediate_sample_weights_(batch_size_ * ceil_divide(N_, WEIGHT_THREADS_PER_BLOCK)),
      d_sample_noise_(round_up_even(num_target_mols_ * num_proposals_per_move_)),
      d_sampling_intermediate_(num_target_mols_ * batch_size_), d_translations_(translation_buffer_size),
      d_sample_segments_offsets_(batch_size_ + 1) {

    if (num_proposals_per_move_ <= 0) {
        throw std::runtime_error("proposals per move must be greater than 0");
    }
    if (mol_size_ == 0) {
        throw std::runtime_error("must provide non-empty molecule indices");
    }
    if (num_proposals_per_move_ % batch_size_ != 0) {
        throw std::runtime_error("num_proposals_per_move must be a multiple of batch size");
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
    if (d_translations_.length / BD_TRANSLATIONS_PER_STEP_XYZW !=
        this->d_quaternions_.length / this->QUATERNIONS_PER_STEP) {
        throw std::runtime_error("bug in the code: buffers with random values don't match in batch size");
    }

    // Set the stream for the generators
    curandErrchk(curandSetStream(cr_rng_quat_, stream));
    curandErrchk(curandSetStream(cr_rng_translations_, stream));
    curandErrchk(curandSetStream(cr_rng_samples_, stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    this->compute_initial_weights(N, d_coords, d_box, stream);

    // All of the noise is generated upfront
    curandErrchk(templateCurandNormal(cr_rng_quat_, d_quaternions_.data, d_quaternions_.length, 0.0, 1.0));
    // The d_translation_ buffer contains uniform noise over [0, 1] containing [x,y,z,w] where [x,y,z] are a random
    // translation and w is used in the metropolis-hastings check
    curandErrchk(templateCurandUniform(cr_rng_translations_, d_translations_.data, d_translations_.length));
    curandErrchk(templateCurandUniform(cr_rng_samples_, d_sample_noise_.data, d_sample_noise_.length));
    for (int step = 0; step < steps_per_move_; step++) {
        // Run only after the first pass, to maintain meaningful `log_probability_host` values
        if (step > 0) {
            // Run a separate kernel to replace the before log probs and weights with the after if accepted a move
            // Need the weights to sample a value and the log probs are just because they aren't expensive to copy
            k_store_accepted_log_probability<RealType><<<1, tpb, 0>>>(
                num_target_mols_,
                d_translations_.data + (step * BD_TRANSLATIONS_PER_STEP_XYZW * batch_size_) +
                    (BD_TRANSLATIONS_PER_STEP_XYZW - 1), // Offset to get the last value for the acceptance criteria
                d_lse_max_before_.data,
                d_lse_exp_sum_before_.data,
                d_lse_max_after_.data,
                d_lse_exp_sum_after_.data,
                d_log_weights_before_.data,
                d_log_weights_after_.data);
            gpuErrchk(cudaPeekAtLastError());
        }

        gpuErrchk(cudaMemcpyAsync(
            d_log_weights_after_.data,
            d_log_weights_before_.data,
            d_log_weights_after_.size(),
            cudaMemcpyDeviceToDevice,
            stream));

        // We only ever sample a single molecule
        sampler_.sample_given_noise_device(
            num_target_mols_ * batch_size_,
            batch_size_,
            d_sample_segments_offsets_.data,
            d_log_weights_before_.data,
            d_sample_noise_.data + (step * num_target_mols_ * batch_size_),
            d_sampling_intermediate_.data,
            d_samples_.data,
            stream);

        // Don't move translations into computation of the incremental, as different translations can be used
        // by different bias deletion movers (such as targeted insertion)
        // scale the translations as they are between [0, 1]
        this->compute_incremental_weights(
            N,
            true,
            d_box,
            d_coords,
            this->d_quaternions_.data + (step * QUATERNIONS_PER_STEP * batch_size_),
            this->d_translations_.data + (step * BD_TRANSLATIONS_PER_STEP_XYZW * batch_size_),
            stream);

        logsumexp_.sum_device(
            num_target_mols_ * batch_size_,
            batch_size_,
            d_sample_segments_offsets_.data,
            d_log_weights_after_.data,
            d_lse_max_after_.data,
            d_lse_exp_sum_after_.data,
            stream);

        k_attempt_exchange_move<RealType><<<1, 1, 0, stream>>>(
            N,
            d_translations_.data + (step * BD_TRANSLATIONS_PER_STEP_XYZW * batch_size_) +
                (BD_TRANSLATIONS_PER_STEP_XYZW - 1), // Offset to get the last value for the acceptance criteria
            d_lse_max_before_.data,
            d_lse_exp_sum_before_.data,
            d_lse_max_after_.data,
            d_lse_exp_sum_after_.data,
            d_target_mol_offsets_.data,
            d_samples_.data,
            d_intermediate_coords_.data,
            d_coords,
            d_num_accepted_.data);
        gpuErrchk(cudaPeekAtLastError());
        num_attempted_++;
    }
}

template <typename RealType>
void BDExchangeMove<RealType>::compute_initial_weights(
    const int N, double *d_coords, double *d_box, cudaStream_t stream) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int mol_blocks = ceil_divide(num_target_mols_, tpb);
    mol_potential_.mol_energies_device(
        N,
        num_target_mols_,
        d_coords,
        d_params_.data,
        d_box,
        d_mol_energy_buffer_.data, // Don't need to zero, will be overridden
        stream);

    // Don't need to normalize to sample
    k_compute_log_weights_from_energies<RealType><<<mol_blocks, tpb, 0, stream>>>(
        num_target_mols_, beta_, d_mol_energy_buffer_.data, d_log_weights_before_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Compute logsumexp of energies once upfront to get log probabilities
    logsumexp_.sum_device(
        num_target_mols_,
        1,
        d_sample_segments_offsets_.data,
        d_log_weights_before_.data,
        d_lse_max_before_.data,
        d_lse_exp_sum_before_.data,
        stream);
}

template <typename RealType>
void BDExchangeMove<RealType>::compute_incremental_weights(
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
        batch_size_,
        mol_size_,
        d_samples_.data,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_target_mol_atoms_.data,
        d_target_mol_offsets_.data);
    gpuErrchk(cudaPeekAtLastError());

    if (scale) {
        k_rotate_and_translate_mols<RealType, true><<<ceil_divide(batch_size_, tpb), tpb, 0, stream>>>(
            batch_size_,
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
            batch_size_,
            d_coords,
            d_box,
            d_samples_.data,
            d_target_mol_offsets_.data,
            d_quaternions,
            d_translations,
            d_intermediate_coords_.data);
        gpuErrchk(cudaPeekAtLastError());
    }

    k_atom_by_atom_energies<<<atom_by_atom_grid, tpb, 0, stream>>>(
        N,
        mol_size_,
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
    // we later call k_set_sampled_weight to set the weight of the sampled mol
    k_adjust_weights<RealType, true><<<ceil_divide(num_target_mols_, tpb), tpb, 0, stream>>>(
        N,
        num_target_mols_,
        mol_size_,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_sample_per_atom_energy_buffer_.data,
        beta_, // 1 / kT
        d_log_weights_after_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_atom_by_atom_energies<<<atom_by_atom_grid, tpb, 0, stream>>>(
        N,
        mol_size_,
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
    k_adjust_weights<RealType, false><<<ceil_divide(num_target_mols_, tpb), tpb, 0, stream>>>(
        N,
        num_target_mols_,
        mol_size_,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_sample_per_atom_energy_buffer_.data,
        beta_, // 1 / kT
        d_log_weights_after_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Set the sampled weight to be the correct value
    k_set_sampled_weight_block<RealType, WEIGHT_THREADS_PER_BLOCK>
        <<<static_cast<int>(d_intermediate_sample_weights_.length), WEIGHT_THREADS_PER_BLOCK, 0, stream>>>(
            N,
            mol_size_,
            d_target_mol_atoms_.data,
            d_sample_per_atom_energy_buffer_.data,
            beta_, // 1 / kT
            d_intermediate_sample_weights_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_set_sampled_weight_reduce<RealType, WEIGHT_THREADS_PER_BLOCK><<<1, WEIGHT_THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<int>(d_intermediate_sample_weights_.length), // Number of intermediates
        d_samples_.data,                                         // where to set the value
        d_intermediate_sample_weights_.data,                     // intermediate fixed point weights
        d_log_weights_after_.data);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType> double BDExchangeMove<RealType>::raw_log_probability_host() {
    std::vector<RealType> h_log_exp_before(2);
    std::vector<RealType> h_log_exp_after(2);
    d_lse_max_before_.copy_to(&h_log_exp_before[0]);
    d_lse_exp_sum_before_.copy_to(&h_log_exp_before[1]);
    d_lse_max_after_.copy_to(&h_log_exp_after[0]);
    d_lse_exp_sum_after_.copy_to(&h_log_exp_after[1]);

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final(h_log_exp_before[0], h_log_exp_before[1]));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final(h_log_exp_after[0], h_log_exp_after[1]));

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
