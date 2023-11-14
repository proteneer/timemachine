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
    : N_(N), mol_size_(target_mols[0].size()), proposals_per_move_(proposals_per_move),
      num_target_mols_(target_mols.size()), nb_beta_(static_cast<RealType>(nb_beta)),
      beta_(static_cast<RealType>(1.0 / (BOLTZ * temperature))),
      cutoff_squared_(static_cast<RealType>(cutoff * cutoff)), radius_(static_cast<RealType>(radius)),
      target_volume_(static_cast<RealType>((4 / 3) * M_PI * pow(radius_, 3))), num_attempted_(0),
      mol_potential_(N, target_mols, nb_beta, cutoff), sampler_(num_target_mols_, seed), logsumexp_(num_target_mols_),
      d_intermediate_coords_(N * 3), d_params_(params), d_mol_energy_buffer_(num_target_mols_),
      d_sample_per_atom_energy_buffer_(mol_size_ * N), d_atom_idxs_(get_atom_indices(target_mols)),
      d_mol_offsets_(get_mol_offsets(target_mols)), d_log_weights_before_(num_target_mols_),
      d_log_weights_after_(num_target_mols_), d_log_sum_exp_before_(2), d_log_sum_exp_after_(2),
      d_samples_(NUM_SAMPLES), d_quaternions_(round_up_even(4)), d_acceptance_(round_up_even(2)), d_num_accepted_(1),
      d_target_mol_atoms_(mol_size_), d_target_mol_offsets_(num_target_mols_ + 1),
      d_rand_states_(DEFAULT_THREADS_PER_BLOCK), d_inner_mols_count_(1), d_inner_mols_(num_target_mols_),
      d_outer_mols_count_(1), d_outer_mols_(num_target_mols_), d_center_(3), d_translation_(NUM_SAMPLES * 3),
      d_inner_flag_(1), d_ligand_idxs_(ligand_idxs), d_src_weights_(num_target_mols_),
      d_dest_weights_(num_target_mols_), p_inner_count_(1), p_inner_flag_(1) {

    if (proposals_per_move_ <= 0) {
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
    gpuErrchk(cudaMemset(d_log_sum_exp_before_.data, 0, d_log_sum_exp_before_.size()));
    gpuErrchk(cudaMemset(d_log_sum_exp_after_.data, 0, d_log_sum_exp_after_.size()));
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));

    // Create event with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&host_copy_event_, cudaEventDisableTiming));

    k_initialize_curand_states<<<1, DEFAULT_THREADS_PER_BLOCK, 0>>>(
        DEFAULT_THREADS_PER_BLOCK, seed, d_rand_states_.data);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType> TIBDExchangeMove<RealType>::~TIBDExchangeMove() {
    curandErrchk(curandDestroyGenerator(cr_rng_));
    gpuErrchk(cudaEventDestroy(host_copy_event_));
}

template <typename RealType>
void TIBDExchangeMove<RealType>::move_device(
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

    dim3 atom_by_atom_grid(ceil_divide(N, tpb), mol_size_, 1);

    k_compute_centroid_of_atoms<RealType><<<ceil_divide(d_ligand_idxs_.length, tpb), tpb, 0, stream>>>(
        static_cast<int>(d_ligand_idxs_.length), d_ligand_idxs_.data, d_coords, d_center_.data);
    gpuErrchk(cudaPeekAtLastError());

    const int num_samples = NUM_SAMPLES;
    for (int move = 0; move < proposals_per_move_; move++) {
        // Run only after the first pass, to maintain meaningful `log_probability_host` values
        if (move > 0) {
            // Run a separate kernel to replace the before log probs and weights with the after if accepted a move
            // Need the weights to sample a value and the log probs are just because they aren't expensive to copy
            k_store_accepted_log_probability<RealType><<<1, tpb, 0>>>(
                num_target_mols_,
                d_acceptance_.data + 1, // Offset to get the last value for the acceptance criteria
                d_log_sum_exp_before_.data,
                d_log_sum_exp_after_.data,
                d_log_weights_before_.data,
                d_log_weights_after_.data);
            gpuErrchk(cudaPeekAtLastError());
        }

        k_split_mols_inner_outer<RealType><<<mol_blocks, tpb, 0, stream>>>(
            num_target_mols_,
            d_atom_idxs_.data,
            d_mol_offsets_.data,
            d_center_.data,
            radius_ * radius_,
            d_coords,
            d_box,
            d_inner_mols_count_.data,
            d_inner_mols_.data,
            d_outer_mols_count_.data,
            d_outer_mols_.data);

        // The d_acceptance_ buffer contains the random value for determining where to insert and whether to accept the move
        curandErrchk(templateCurandUniform(cr_rng_, d_acceptance_.data, d_acceptance_.length));

        k_decide_targeted_move<<<1, 1, 0, stream>>>(
            d_acceptance_.data, d_inner_mols_count_.data, d_outer_mols_count_.data, d_inner_flag_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Copy count and flag to the host, needed to know how many values to look at for
        // sampling and logsumexp
        gpuErrchk(cudaMemcpyAsync(
            p_inner_count_.data, d_inner_mols_count_.data, d_inner_mols_count_.size(), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(
            p_inner_flag_.data, d_inner_flag_.data, d_inner_flag_.size(), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaEventRecord(host_copy_event_, stream));

        k_generate_translations_within_or_outside_a_sphere<<<ceil_divide(num_samples, tpb), tpb, 0, stream>>>(
            num_samples, d_box, d_center_.data, d_inner_flag_.data, radius_, d_rand_states_.data, d_translation_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Copy the before log weights to the after weights, we will adjust the after weights incrementally
        gpuErrchk(cudaMemcpyAsync(
            d_log_weights_after_.data,
            d_log_weights_before_.data,
            d_log_weights_after_.size(),
            cudaMemcpyDeviceToDevice,
            stream));
        // Make a copy of the coordinates
        gpuErrchk(cudaMemcpyAsync(
            d_intermediate_coords_.data, d_coords, d_intermediate_coords_.size(), cudaMemcpyDeviceToDevice, stream));

        // Quaternions generated from normal noise generate uniform rotations
        curandErrchk(templateCurandNormal(cr_rng_, d_quaternions_.data, d_quaternions_.length, 0.0, 1.0));

        k_separate_weights_for_targeted<RealType><<<mol_blocks, tpb, 0, stream>>>(
            num_target_mols_,
            d_inner_flag_.data,
            d_inner_mols_count_.data,
            d_outer_mols_count_.data,
            d_inner_mols_.data,
            d_outer_mols_.data,
            d_log_weights_before_.data,
            d_src_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaEventSynchronize(host_copy_event_));
        int inner_count = p_inner_count_.data[0];
        int inner_flag = p_inner_flag_.data[0];
        // Inner flag == 1 indicates that we are moving from outside volume to inner volume
        int src_count = inner_flag == 0 ? inner_count : num_target_mols_ - inner_count;
        int dest_count = num_target_mols_ - src_count;

        logsumexp_.sum_device(src_count, d_src_weights_.data, d_log_sum_exp_before_.data, stream);

        // Past here pretty much 'normal'
        sampler_.sample_device(src_count, num_samples, d_src_weights_.data, d_samples_.data, stream);

        k_setup_sample_atoms<<<ceil_divide(num_samples, tpb), tpb, 0, stream>>>(
            num_samples,
            mol_size_,
            d_samples_.data,
            d_atom_idxs_.data,
            d_mol_offsets_.data,
            d_target_mol_atoms_.data,
            d_target_mol_offsets_.data);
        gpuErrchk(cudaPeekAtLastError());

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

        // Don't scale translations by box
        k_rotate_and_translate_mols<RealType, false><<<ceil_divide(num_samples, tpb), tpb, 0, stream>>>(
            num_samples,
            d_coords,
            d_box,
            d_samples_.data,
            d_target_mol_offsets_.data,
            d_quaternions_.data,
            d_translation_.data,
            d_intermediate_coords_.data);
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
        k_set_sampled_weight<RealType, WEIGHT_THREADS_PER_BLOCK><<<1, WEIGHT_THREADS_PER_BLOCK, 0, stream>>>(
            N,
            mol_size_,
            num_samples,
            d_samples_.data,
            d_target_mol_atoms_.data,
            d_mol_offsets_.data,
            d_sample_per_atom_energy_buffer_.data,
            beta_, // 1 / kT
            d_log_weights_after_.data);
        gpuErrchk(cudaPeekAtLastError());

        k_setup_destination_weights_for_targeted<RealType><<<mol_blocks, tpb, 0, stream>>>(
            num_target_mols_,
            num_samples,
            d_samples_.data,
            d_inner_flag_.data,
            d_inner_mols_count_.data,
            d_outer_mols_count_.data,
            d_inner_mols_.data,
            d_outer_mols_.data,
            d_log_weights_after_.data,
            d_dest_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Add one to the destination count, as we just moved a mol there
        logsumexp_.sum_device(dest_count + 1, d_dest_weights_.data, d_log_sum_exp_after_.data, stream);

        k_attempt_exchange_move<RealType><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N,
            d_acceptance_.data + 1, // Offset to get the last value for the acceptance criteria
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
TIBDExchangeMove<RealType>::move_host(const int N, const double *h_coords, const double *h_box) {

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
    d_log_sum_exp_before_.copy_to(&h_log_exp_before[0]);
    d_log_sum_exp_after_.copy_to(&h_log_exp_after[0]);

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_before[0]));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final(&h_log_exp_after[0]));

    return min(static_cast<double>(before_log_prob - after_log_prob), 0.0);
}

template <typename RealType> size_t TIBDExchangeMove<RealType>::n_accepted() const {
    size_t h_accepted;
    d_num_accepted_.copy_to(&h_accepted);
    return h_accepted;
}

template class TIBDExchangeMove<float>;
template class TIBDExchangeMove<double>;

} // namespace timemachine
