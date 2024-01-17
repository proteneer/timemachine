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
    const int proposals_per_move,
    const int interval)
    : Mover(interval), N_(N), mol_size_(target_mols[0].size()), proposals_per_move_(proposals_per_move),
      num_target_mols_(target_mols.size()), nb_beta_(static_cast<RealType>(nb_beta)),
      beta_(static_cast<RealType>(1.0 / (BOLTZ * temperature))),
      cutoff_squared_(static_cast<RealType>(cutoff * cutoff)), num_attempted_(0),
      mol_potential_(N, target_mols, nb_beta, cutoff), sampler_(num_target_mols_, seed), logsumexp_(num_target_mols_),
      d_intermediate_coords_(N * 3), d_params_(params), d_mol_energy_buffer_(num_target_mols_),
      d_sample_per_atom_energy_buffer_(mol_size_ * N), d_atom_idxs_(get_atom_indices(target_mols)),
      d_mol_offsets_(get_mol_offsets(target_mols)), d_log_weights_before_(num_target_mols_),
      d_log_weights_after_(num_target_mols_), d_log_sum_exp_before_(2), d_log_sum_exp_after_(2), d_samples_(1),
      d_quaternions_(round_up_even(QUATERNIONS_PER_STEP * proposals_per_move_)), d_num_accepted_(1),
      d_target_mol_atoms_(mol_size_), d_target_mol_offsets_(num_target_mols_ + 1),
      d_intermediate_sample_weights_(ceil_divide(N_, WEIGHT_THREADS_PER_BLOCK)),
      d_sample_noise_(round_up_even(num_target_mols_ * proposals_per_move_)),
      d_translations_(round_up_even(BD_TRANSLATIONS_PER_STEP_XYZW * proposals_per_move_)) {

    if (proposals_per_move_ <= 0) {
        throw std::runtime_error("proposals per move must be greater than 0");
    }
    if (mol_size_ == 0) {
        throw std::runtime_error("must provide non-empty molecule indices");
    }
    if (d_translations_.length / BD_TRANSLATIONS_PER_STEP_XYZW !=
        this->d_quaternions_.length / this->QUATERNIONS_PER_STEP) {
        throw std::runtime_error("bug in the code: buffers with random values don't match in batch size");
    }
    verify_mols_contiguous(target_mols);
    for (int i = 0; i < target_mols.size(); i++) {
        if (target_mols[i].size() != mol_size_) {
            throw std::runtime_error("only support running with mols with constant size, got mixed sizes");
        }
    }
    // Clear out the logsumexp values so the log probability starts off as zero
    gpuErrchk(cudaMemset(d_log_sum_exp_before_.data, 0, d_log_sum_exp_before_.size()));
    gpuErrchk(cudaMemset(d_log_sum_exp_after_.data, 0, d_log_sum_exp_after_.size()));
    gpuErrchk(cudaMemset(d_num_accepted_.data, 0, d_num_accepted_.size()));
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
}

template <typename RealType> BDExchangeMove<RealType>::~BDExchangeMove() {
    curandErrchk(curandDestroyGenerator(cr_rng_));
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

    // Set the stream for the generator
    curandErrchk(curandSetStream(cr_rng_, stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    this->compute_initial_weights(N, d_coords, d_box, stream);

    // All of the noise is generated upfront
    curandErrchk(templateCurandNormal(cr_rng_, d_quaternions_.data, d_quaternions_.length, 0.0, 1.0));
    // The d_translation_ buffer contains uniform noise over [0, 1] containing [x,y,z,w] where [x,y,z] are a random
    // translation and w is used in the metropolis-hastings check
    curandErrchk(templateCurandUniform(cr_rng_, d_translations_.data, d_translations_.length));
    curandErrchk(templateCurandUniform(cr_rng_, d_sample_noise_.data, d_sample_noise_.length));
    for (int move = 0; move < proposals_per_move_; move++) {
        // Run only after the first pass, to maintain meaningful `log_probability_host` values
        if (move > 0) {
            // Run a separate kernel to replace the before log probs and weights with the after if accepted a move
            // Need the weights to sample a value and the log probs are just because they aren't expensive to copy
            k_store_accepted_log_probability<RealType><<<1, tpb, 0>>>(
                num_target_mols_,
                d_translations_.data + (move * BD_TRANSLATIONS_PER_STEP_XYZW) +
                    3, // Offset to get the last value for the acceptance criteria
                d_log_sum_exp_before_.data,
                d_log_sum_exp_after_.data,
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
            num_target_mols_,
            1,
            d_log_weights_before_.data,
            d_sample_noise_.data + (move * num_target_mols_),
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
            this->d_quaternions_.data + (move * QUATERNIONS_PER_STEP),
            this->d_translations_.data + (move * BD_TRANSLATIONS_PER_STEP_XYZW),
            stream);

        logsumexp_.sum_device(num_target_mols_, d_log_weights_after_.data, d_log_sum_exp_after_.data, stream);

        k_attempt_exchange_move<RealType><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N,
            d_translations_.data + (move * BD_TRANSLATIONS_PER_STEP_XYZW) +
                3, // Offset to get the last value for the acceptance criteria
            d_log_sum_exp_before_.data,
            d_log_sum_exp_after_.data,
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
    logsumexp_.sum_device(num_target_mols_, d_log_weights_before_.data, d_log_sum_exp_before_.data, stream);
}

template <typename RealType>
void BDExchangeMove<RealType>::compute_incremental_weights(
    const int N,
    const bool scale,
    const double *d_box,            // [3, 3]
    const double *d_coords,         // [N, 3]
    const RealType *d_quaternions,  // [4]
    const RealType *d_translations, // [3]
    cudaStream_t stream) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    dim3 atom_by_atom_grid(ceil_divide(N, tpb), mol_size_, 1);

    // Make a copy of the coordinates
    gpuErrchk(cudaMemcpyAsync(
        d_intermediate_coords_.data, d_coords, d_intermediate_coords_.size(), cudaMemcpyDeviceToDevice, stream));

    // Only support sampling a single mol at this time, so only one block
    k_setup_sample_atoms<<<1, tpb, 0, stream>>>(
        mol_size_,
        d_samples_.data,
        d_atom_idxs_.data,
        d_mol_offsets_.data,
        d_target_mol_atoms_.data,
        d_target_mol_offsets_.data);
    gpuErrchk(cudaPeekAtLastError());

    if (scale) {
        k_rotate_and_translate_mols<RealType, true><<<1, tpb, 0, stream>>>(
            1,
            d_coords,
            d_box,
            d_samples_.data,
            d_target_mol_offsets_.data,
            d_quaternions,
            d_translations,
            d_intermediate_coords_.data);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        k_rotate_and_translate_mols<RealType, false><<<1, tpb, 0, stream>>>(
            1,
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
    d_log_sum_exp_before_.copy_to(&h_log_exp_before[0]);
    d_log_sum_exp_after_.copy_to(&h_log_exp_after[0]);

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_before[0]));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_after[0]));

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
