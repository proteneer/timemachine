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

namespace timemachine {

// NOISE_PER_STEP is the uniform generated per step that is used for deciding the targeted move as well as the acceptance
// in the metropolis hasting check.
static const int NOISE_PER_STEP = 2;

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
    const int proposals_per_move,
    const int interval)
    : BDExchangeMove<RealType>(
          N, target_mols, params, temperature, nb_beta, cutoff, seed, proposals_per_move, interval),
      radius_(static_cast<RealType>(radius)), inner_volume_(static_cast<RealType>((4.0 / 3.0) * M_PI * pow(radius, 3))),
      quaternion_offset_(0), d_rand_states_(DEFAULT_THREADS_PER_BLOCK), d_inner_mols_count_(1),
      d_identify_indices_(this->num_target_mols_), d_partitioned_indices_(this->num_target_mols_),
      d_temp_storage_buffer_(0), d_center_(3),
      d_uniform_noise_buffer_(round_up_even(NOISE_PER_STEP * this->RANDOM_BATCH_SIZE)), d_targeting_inner_vol_(1),
      d_ligand_idxs_(ligand_idxs), d_src_weights_(this->num_target_mols_), d_dest_weights_(this->num_target_mols_),
      d_inner_flags_(this->num_target_mols_), d_box_volume_(1), p_inner_count_(1), p_targeting_inner_vol_(1) {

    if (radius <= 0.0) {
        throw std::runtime_error("radius must be greater than 0.0");
    }
    if (d_uniform_noise_buffer_.length / NOISE_PER_STEP != this->d_quaternions_.length / this->QUATERNIONS_PER_STEP) {
        throw std::runtime_error("bug in the code: buffers with random values don't match in batch size");
    }

    // Create event with timings disabled as timings slow down events
    gpuErrchk(cudaEventCreateWithFlags(&host_copy_event_, cudaEventDisableTiming));

    k_initialize_curand_states<<<1, DEFAULT_THREADS_PER_BLOCK, 0>>>(
        DEFAULT_THREADS_PER_BLOCK, seed, d_rand_states_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_arange<<<ceil_divide(this->num_target_mols_, DEFAULT_THREADS_PER_BLOCK), DEFAULT_THREADS_PER_BLOCK, 0>>>(
        this->num_target_mols_, d_identify_indices_.data, 0);
    gpuErrchk(cudaPeekAtLastError());

    // Setup buffer for doing the flagged partition
    gpuErrchk(cub::DevicePartition::Flagged(
        nullptr,
        temp_storage_bytes_,
        d_identify_indices_.data,
        d_inner_flags_.data,
        d_partitioned_indices_.data,
        d_inner_mols_count_.data,
        this->num_target_mols_));
    // Allocate char as temp_storage_bytes_ is in raw bytes and the type doesn't matter in practice.
    // Equivalent to DeviceBuffer<int> buf(temp_storage_bytes_ / sizeof(int))
    d_temp_storage_buffer_.realloc(temp_storage_bytes_);

    // Set the offset to the length of the random vector to ensure noise is triggered on first step
    this->noise_offset_ = this->d_uniform_noise_buffer_.length;
}

template <typename RealType> TIBDExchangeMove<RealType>::~TIBDExchangeMove() {
    gpuErrchk(cudaEventDestroy(host_copy_event_));
}

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

    // Set the stream for the generator
    curandErrchk(curandSetStream(this->cr_rng_, stream));

    this->compute_initial_weights(N, d_coords, d_box, stream);

    // Copy the before log weights to the after weights, we will adjust incrementally afterwards
    gpuErrchk(cudaMemcpyAsync(
        this->d_log_weights_after_.data,
        this->d_log_weights_before_.data,
        this->d_log_weights_after_.size(),
        cudaMemcpyDeviceToDevice,
        stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int mol_blocks = ceil_divide(this->num_target_mols_, tpb);

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

    for (int move = 0; move < this->proposals_per_move_; move++) {
        if (this->noise_offset_ >= this->d_uniform_noise_buffer_.length) {
            // reset the noise to zero and generate more noise
            this->noise_offset_ = 0;
            quaternion_offset_ = 0;
            curandErrchk(templateCurandUniform(
                this->cr_rng_, this->d_uniform_noise_buffer_.data, this->d_uniform_noise_buffer_.length));
            curandErrchk(
                templateCurandNormal(this->cr_rng_, this->d_quaternions_.data, this->d_quaternions_.length, 0.0, 1.0));
        }

        gpuErrchk(cub::DevicePartition::Flagged(
            d_temp_storage_buffer_.data,
            temp_storage_bytes_,
            d_identify_indices_.data,
            d_inner_flags_.data,
            d_partitioned_indices_.data,
            d_inner_mols_count_.data,
            this->num_target_mols_,
            stream));

        k_decide_targeted_move<<<1, 1, 0, stream>>>(
            this->num_target_mols_,
            this->d_uniform_noise_buffer_.data + this->noise_offset_,
            d_inner_mols_count_.data,
            d_targeting_inner_vol_.data);
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

        k_generate_translations_within_or_outside_a_sphere<<<1, tpb, 0, stream>>>(
            1,
            d_box,
            d_center_.data,
            d_targeting_inner_vol_.data,
            radius_,
            d_rand_states_.data,
            this->d_translations_.data);
        gpuErrchk(cudaPeekAtLastError());

        k_separate_weights_for_targeted<RealType><<<mol_blocks, tpb, 0, stream>>>(
            this->num_target_mols_,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_partitioned_indices_.data,
            this->d_log_weights_before_.data,
            d_src_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaEventSynchronize(host_copy_event_));
        int inner_count = p_inner_count_.data[0];

        // targeting_inner_vol == 1 indicates that we are targeting the inner volume, starting from the outer mols
        int targeting_inner_vol = p_targeting_inner_vol_.data[0];
        int src_count = targeting_inner_vol == 0 ? inner_count : this->num_target_mols_ - inner_count;
        int dest_count = this->num_target_mols_ - src_count;

        this->logsumexp_.sum_device(src_count, d_src_weights_.data, this->d_log_sum_exp_before_.data, stream);

        // Only sample one mol
        this->sampler_.sample_device(src_count, 1, d_src_weights_.data, this->d_samples_.data, stream);

        // Selected an index from the src weights, need to remap the samples idx to the mol indices
        k_adjust_sample_idx<<<1, 1, 0, stream>>>(
            d_targeting_inner_vol_.data, d_inner_mols_count_.data, d_partitioned_indices_.data, this->d_samples_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Don't move translations into computation of the incremental, as different translations can be used
        // by different bias deletion movers (such as targeted insertion)
        // Don't scale the translations as they are computed to be within the region
        this->compute_incremental_weights(
            N, false, d_coords, d_box, this->d_quaternions_.data + quaternion_offset_, stream);

        k_setup_destination_weights_for_targeted<RealType><<<mol_blocks, tpb, 0, stream>>>(
            this->num_target_mols_,
            this->d_samples_.data,
            d_targeting_inner_vol_.data,
            d_inner_mols_count_.data,
            d_partitioned_indices_.data,
            this->d_log_weights_after_.data,
            d_dest_weights_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Add one to the destination count, as we just moved a mol there
        this->logsumexp_.sum_device(dest_count + 1, d_dest_weights_.data, this->d_log_sum_exp_after_.data, stream);

        k_attempt_exchange_move_targeted<RealType><<<ceil_divide(N, tpb), tpb, 0, stream>>>(
            N,
            this->num_target_mols_,
            d_targeting_inner_vol_.data,
            d_box_volume_.data,
            inner_volume_,
            // Offset to get the last value for the acceptance criteria
            this->d_uniform_noise_buffer_.data + this->noise_offset_ + 1,
            this->d_samples_.data,
            this->d_log_sum_exp_before_.data,
            this->d_log_sum_exp_after_.data,
            this->d_intermediate_coords_.data,
            d_coords,
            this->d_log_weights_before_.data,
            this->d_log_weights_after_.data,
            d_inner_flags_.data,
            this->d_num_accepted_.data);
        gpuErrchk(cudaPeekAtLastError());
        this->num_attempted_++;
        this->noise_offset_ += NOISE_PER_STEP;
        quaternion_offset_ += this->QUATERNIONS_PER_STEP;
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

    this->move(N, d_coords.data, d_box.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<double> out_coords(d_coords.length);
    d_coords.copy_to(&out_coords[0]);

    std::vector<double> out_box(d_box.length);
    d_box.copy_to(&out_box[0]);

    return std::array<std::vector<double>, 2>({out_coords, out_box});
}

template <typename RealType> double TIBDExchangeMove<RealType>::raw_log_probability_host() {
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

    return static_cast<double>(before_log_prob - after_log_prob + log_vol_prob);
}

template <typename RealType> double TIBDExchangeMove<RealType>::log_probability_host() {
    return min(raw_log_probability_host(), 0.0);
}

template class TIBDExchangeMove<float>;
template class TIBDExchangeMove<double>;

} // namespace timemachine
