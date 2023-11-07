#include "bd_exchange_move.hpp"

#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_exchange.cuh"
#include "kernels/k_probability.cuh"
#include "kernels/k_rotations.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"

namespace timemachine {

template <typename RealType>
BDExchangeMove<RealType>::BDExchangeMove(
    const int N,
    const std::vector<std::vector<int>> &target_mols,
    const std::vector<double> &params,
    const double temperature,
    const double nb_beta,
    const double cutoff,
    const int seed,
    const int proposals_per_move)
    : N_(N), proposals_per_move_(proposals_per_move), num_target_mols_(target_mols.size()),
      beta_(static_cast<RealType>(1.0 / (BOLTZ * temperature))), mol_potential_(N, target_mols, nb_beta, cutoff),
      sampler_(num_target_mols_, seed), logsumexp_(N), d_intermediate_coords_(N * 3), d_params_(params.size()),
      d_mol_energy_buffer_(num_target_mols_), d_mol_offsets_(get_mol_offsets(target_mols).size()),
      d_log_weights_before_(num_target_mols_), d_log_weights_after_(num_target_mols_),
      d_log_probabilities_before_(num_target_mols_), d_log_probabilities_after_(num_target_mols_),
      d_log_sum_exp_before_(2), d_log_sum_exp_after_(2), d_samples_(1), d_quaternions_(round_up_even(4)),
      d_translations_(round_up_even(4)), d_num_accepted_(1), num_attempted_(0) {
    d_params_.copy_from(&params[0]);
    d_mol_offsets_.copy_from(&get_mol_offsets(target_mols)[0]);

    // Clear out the logsumexp values so the log probability starts off as zero
    gpuErrchk(cudaMemset(d_log_sum_exp_before_.data, 0, d_log_sum_exp_before_.size()));
    gpuErrchk(cudaMemset(d_log_sum_exp_after_.data, 0, d_log_sum_exp_after_.size()));
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
}

template <typename RealType> BDExchangeMove<RealType>::~BDExchangeMove() {
    curandErrchk(curandDestroyGenerator(cr_rng_));
}

template <typename RealType>
void BDExchangeMove<RealType>::move_device(
    const int N,
    double *d_coords, // [N, 3]
    double *d_box,    // [3, 3]
    cudaStream_t stream) {

    if (N != N_) {
        throw std::runtime_error("N != N_");
    }

    // Set the stream for the generator
    curandErrchk(curandSetStream(cr_rng_, stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int mol_blocks = ceil_divide(num_target_mols_, tpb);
    // Compute logsumexp of energies once upfront to get log probabilities
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

    logsumexp_.sum_device(num_target_mols_, d_log_weights_before_.data, d_log_sum_exp_before_.data, stream);

    const int num_samples = 1;
    for (int move = 0; move < proposals_per_move_; move++) {
        // Run only after the first pass, to maintain meaningful `log_probability_host` values
        if (move > 0) {
            // Run a separate kernel to replace the before log probs and weights with the after if accepted a move
            // Need the weights to sample a value and the log probs are just because they aren't expensive to copy
            k_store_accepted_log_probability<RealType><<<1, tpb, 0>>>(
                num_target_mols_,
                d_translations_.data + 3, // Offset to get the last value for the acceptance criteria
                d_log_sum_exp_before_.data,
                d_log_sum_exp_after_.data,
                d_log_weights_before_.data,
                d_log_weights_after_.data);
            gpuErrchk(cudaPeekAtLastError());
        }
        // Make a copy of the coordinates
        gpuErrchk(cudaMemcpyAsync(
            d_intermediate_coords_.data, d_coords, d_intermediate_coords_.size(), cudaMemcpyDeviceToDevice, stream));

        // Quaternions generated from normal noise generate uniform rotations
        curandErrchk(templateCurandNormal(cr_rng_, d_quaternions_.data, d_quaternions_.length, 0.0, 1.0));
        // The d_translation_ buffer is [x,y,z,w] where [x,y,z] are a random translation and w is used for acceptance
        curandErrchk(templateCurandUniform(cr_rng_, d_translations_.data, d_translations_.length));

        sampler_.sample_device(num_target_mols_, num_samples, d_log_weights_before_.data, d_samples_.data, stream);
        k_rotate_and_translate_mols<RealType><<<ceil_divide(num_samples, tpb), tpb, 0, stream>>>(
            num_samples,
            d_coords,
            d_box,
            d_samples_.data,
            d_mol_offsets_.data,
            d_quaternions_.data,
            d_translations_.data,
            d_intermediate_coords_.data);
        gpuErrchk(cudaPeekAtLastError());

        mol_potential_.mol_energies_device(
            N,
            num_target_mols_,
            d_intermediate_coords_.data, // Use the moved coords
            d_params_.data,
            d_box,
            d_mol_energy_buffer_.data, // Don't need to zero, will be overridden
            stream);

        k_compute_log_weights_from_energies<RealType><<<mol_blocks, tpb, 0, stream>>>(
            num_target_mols_, beta_, d_mol_energy_buffer_.data, d_log_weights_after_.data);
        gpuErrchk(cudaPeekAtLastError());

        logsumexp_.sum_device(num_target_mols_, d_log_weights_after_.data, d_log_sum_exp_after_.data, stream);

        k_attempt_exchange_move<RealType><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N,
            d_translations_.data + 3, // Offset to get the last value for the acceptance criteria
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
std::array<std::vector<double>, 2>
BDExchangeMove<RealType>::move_host(const int N, const double *h_coords, const double *h_box) {

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

template <typename RealType> double BDExchangeMove<RealType>::log_probability_host() {
    std::vector<RealType> h_log_exp_before(2);
    std::vector<RealType> h_log_exp_after(2);
    d_log_sum_exp_before_.copy_to(&h_log_exp_before[0]);
    d_log_sum_exp_after_.copy_to(&h_log_exp_after[0]);

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_before[0]));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_after[0]));

    return min(static_cast<double>(before_log_prob - after_log_prob), 0.0);
}

template <typename RealType> size_t BDExchangeMove<RealType>::n_accepted() const {
    size_t h_accepted;
    d_num_accepted_.copy_to(&h_accepted);
    return h_accepted;
}

template class BDExchangeMove<float>;
template class BDExchangeMove<double>;

} // namespace timemachine
