#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "kernels/k_flat_bottom_bond.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_local_md.cuh"
#include "local_md_potentials.hpp"
#include "math_utils.cuh"
#include "nonbonded_common.hpp"
#include <cub/cub.cuh>
#include <random>
#include <vector>

namespace timemachine {

// Struct representing the CUB < operation
struct LessThan {
    int compare;
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ explicit LessThan(int compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ bool operator()(const int &a) const { return (a < compare); }
};

LocalMDPotentials::LocalMDPotentials(
    const int N, const std::vector<std::shared_ptr<BoundPotential>> &bps, bool freeze_reference, double temperature)
    : freeze_reference(freeze_reference), temperature(temperature), N_(N), temp_storage_bytes_(0), all_potentials_(bps),
      d_restraint_pairs_(N_ * 2), d_bond_params_(N_ * 3), d_probability_buffer_(N_), d_free_idxs_(N_), d_temp_idxs_(N_),
      d_all_pairs_idxs_(N_), d_temp_storage_buffer_(0), d_row_idxs_(N_), d_col_idxs_(N_), p_num_selected_(1),
      d_num_selected_buffer_(1) {

    if (temperature <= 0.0) {
        throw std::runtime_error("temperature must be greater than 0");
    }

    std::vector<std::shared_ptr<BoundPotential>> nonbonded_pots;
    get_nonbonded_all_pair_potentials(bps, nonbonded_pots);

    if (nonbonded_pots.size() > 1) {
        throw std::runtime_error("found multiple NonbondedAllPairs potentials");
    }
    if (nonbonded_pots.size() != 1) {
        throw std::runtime_error("unable to find a NonbondedAllPairs potential");
    }

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    // Only used to reference shared_ptr to potential and for Nonbonded parameters
    // modifications to the BoundPotential has no impact
    nonbonded_bp_ = nonbonded_pots[0];

    // Ensure that we allocate enough space for all potential bonds
    // default_bonds[i * 2 + 0] != default_bonds[i * 2 + 1], so set first value to 0, second to i + 1
    std::vector<int> default_bonds(N_ * 2);
    for (int i = 0; i < N_; i++) {
        default_bonds[i * 2 + 0] = 0;
        default_bonds[i * 2 + 1] = i + 1;
    }
    std::vector<double> default_params(N_ * 3);
    free_restraint_ = std::shared_ptr<FlatBottomBond<float>>(new FlatBottomBond<float>(default_bonds));
    // Construct a bound potential with 0 params
    bound_free_restraint_ = std::shared_ptr<BoundPotential>(new BoundPotential(free_restraint_, default_params));

    // Ensure that the reference idxs start out as all N_
    k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb>>>(N_, d_all_pairs_idxs_.data, N_);
    gpuErrchk(cudaPeekAtLastError());
    num_allpairs_idxs_ = copy_nonbonded_potential_idxs(nonbonded_bp_->potential, N_, d_all_pairs_idxs_.data);

    ixn_group_ =
        construct_ixn_group_potential(N_, nonbonded_bp_->potential, nonbonded_bp_->size, nonbonded_bp_->d_p.data);

    // Add the restraint potential and ixn group potential
    all_potentials_.push_back(bound_free_restraint_);
    all_potentials_.push_back(ixn_group_);
    if (!freeze_reference) {
        frozen_restraint_ = std::shared_ptr<LogFlatBottomBond<float>>(
            new LogFlatBottomBond<float>(default_bonds, 1 / (temperature * BOLTZ)));
        bound_frozen_restraint_ =
            std::shared_ptr<BoundPotential>(new BoundPotential(frozen_restraint_, default_params));
        all_potentials_.push_back(bound_frozen_restraint_);
    }

    gpuErrchk(cub::DevicePartition::If(
        nullptr,
        temp_storage_bytes_,
        d_free_idxs_.data,
        d_row_idxs_.data,
        d_num_selected_buffer_.data,
        N_,
        LessThan(N_)));
    // Allocate char as temp_storage_bytes_ is in raw bytes and the type doesn't matter in practice.
    // Equivalent to DeviceBuffer<int> buf(temp_storage_bytes_ / sizeof(int))
    d_temp_storage_buffer_.realloc(temp_storage_bytes_);

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
};

LocalMDPotentials::~LocalMDPotentials() { curandErrchk(curandDestroyGenerator(cr_rng_)); }

// setup_from_idxs takes a set of idxs and a seed to determine the free particles. Fix the local_idxs to length
// one to ensure the same reference every time, though the seed also handles the probabilities of selecting particles, and it is suggested
// to provide a new seed at each step.
void LocalMDPotentials::setup_from_idxs(
    double *d_x_t,
    double *d_box_t,
    const std::vector<int> &local_idxs,
    const int seed,
    const double radius,
    const double k,
    cudaStream_t stream) {
    curandErrchk(curandSetStream(cr_rng_, stream));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
    // Reset the generator offset to ensure same values for the same seed are produced
    // Simply reseeding does NOT produce identical results
    curandErrchk(curandSetGeneratorOffset(cr_rng_, 0));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    // Set the array to all N, which indicates to ignore that idx
    k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_idxs_.data, N_);
    gpuErrchk(cudaPeekAtLastError());

    // Generate values between (0, 1.0]
    curandErrchk(curandGenerateUniform(cr_rng_, d_probability_buffer_.data, d_probability_buffer_.length));

    std::mt19937 rng;
    rng.seed(seed);
    std::uniform_int_distribution<unsigned int> random_dist(0, local_idxs.size() - 1);

    unsigned int reference_idx = local_idxs[random_dist(rng)];

    const double kBT = BOLTZ * temperature;
    // Select all of the particles that will be free
    k_log_probability_selection<float><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
        N_,
        kBT,
        static_cast<float>(radius),
        static_cast<float>(k),
        reference_idx,
        d_x_t,
        d_box_t,
        d_probability_buffer_.data,
        d_free_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());

    this->_setup_free_idxs_given_reference_idx(reference_idx, radius, k, stream);
}

// setup_from_selection takes a set of idxs, flat-bottom restraint parameters (radius, k)
// assumes selection_idxs are sampled based on exp(-beta U_flat_bottom(distance_to_reference, radius, k))
// (or that the user is otherwise accounting for selection probabilities)
void LocalMDPotentials::setup_from_selection(
    const int reference_idx,
    const std::vector<int> &selection_idxs,
    const double radius,
    const double k,
    const cudaStream_t stream) {

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    // Set the array to all N, which indicates to ignore that idx
    k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_idxs_.data, N_);
    gpuErrchk(cudaPeekAtLastError());

    k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_row_idxs_.data, N_);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpyAsync(
        d_row_idxs_.data,
        &selection_idxs[0],
        selection_idxs.size() * sizeof(*d_row_idxs_.data),
        cudaMemcpyHostToDevice,
        stream));

    // Split out the values from the selection idxs into the indices of the free
    k_unique_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, N_, d_row_idxs_.data, d_free_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());

    this->_setup_free_idxs_given_reference_idx((unsigned int)reference_idx, radius, k, stream);
}

void LocalMDPotentials::_setup_free_idxs_given_reference_idx(
    const unsigned int reference_idx, const double radius, const double k, cudaStream_t stream) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    LessThan select_op(N_);

    if (!freeze_reference) {
        // Remove the reference idx from the column indices, d_free_idxs gets inverted to construct column idxs,
        // and add to the row indices
        k_update_index<<<1, 1, 0, stream>>>(d_free_idxs_.data, reference_idx, reference_idx);
        gpuErrchk(cudaPeekAtLastError());
    }

    int indices_to_remove = N_ - num_allpairs_idxs_;
    unsigned int *d_free_idx_ptr = d_free_idxs_.data;
    // If the atom indices of all pairs isn't all, take intersection with free
    if (indices_to_remove > 0) {
        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_col_idxs_.data, N_);
        gpuErrchk(cudaPeekAtLastError());
        // Spread atom indices out into array with each value at its own index (Val 0 will be at index 0)
        k_unique_indices<<<ceil_divide(num_allpairs_idxs_, tpb), tpb, 0, stream>>>(
            num_allpairs_idxs_, N_, d_all_pairs_idxs_.data, d_col_idxs_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Update the free indices so that only indices that are also in the allpairs indices are considered free
        k_idxs_intersection<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N_, d_col_idxs_.data, d_free_idxs_.data, d_temp_idxs_.data);
        gpuErrchk(cudaPeekAtLastError());
        d_free_idx_ptr = d_temp_idxs_.data;
    }

    // Partition the free idxs into the row idxs
    gpuErrchk(cub::DevicePartition::If(
        d_temp_storage_buffer_.data,
        temp_storage_bytes_,
        d_free_idx_ptr,
        d_row_idxs_.data,
        d_num_selected_buffer_.data,
        N_,
        select_op,
        stream));

    gpuErrchk(cudaMemcpyAsync(
        p_num_selected_.data,
        d_num_selected_buffer_.data,
        1 * sizeof(*p_num_selected_.data),
        cudaMemcpyDeviceToHost,
        stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    // The row indices is all of the free indices, which excludes the reference
    const int num_row_idxs = p_num_selected_.data[0];
    const int num_col_idxs = N_ - num_row_idxs - indices_to_remove;

    if (num_row_idxs == 0) {
        throw std::runtime_error("LocalMDPotentials setup has no free particles selected");
    }

    // The reference particle will always be in the column idxs if the reference is frozen
    if (num_row_idxs == N_ - 1 || (!freeze_reference && num_row_idxs == N_)) {
        fprintf(stderr, "LocalMDPotentials setup has entire system selected\n");
    }

    // Set the nonbonded potential to compute forces of free particles
    set_nonbonded_potential_idxs(nonbonded_bp_->potential, num_row_idxs, d_row_idxs_.data, stream);

    k_construct_bonded_params<<<ceil_divide(num_row_idxs, tpb), tpb, 0, stream>>>(
        num_row_idxs,
        N_,
        reference_idx,
        k,
        0.0,
        radius,
        d_row_idxs_.data,
        d_restraint_pairs_.data,
        d_bond_params_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Setup the flat bottom restraints
    bound_free_restraint_->set_params_device(3 * num_row_idxs, d_bond_params_.data, stream);
    free_restraint_->set_bonds_device(num_row_idxs, d_restraint_pairs_.data, stream);

    // Invert to get column idxs
    k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());

    d_free_idx_ptr = d_free_idxs_.data;
    // If the atom indices of all pairs isn't all, take intersection with frozen
    if (indices_to_remove > 0) {
        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_col_idxs_.data, N_);
        gpuErrchk(cudaPeekAtLastError());
        // Spread atom indices out into array with each value at its own index (Val 0 will be at index 0)
        k_unique_indices<<<ceil_divide(num_allpairs_idxs_, tpb), tpb, 0, stream>>>(
            num_allpairs_idxs_, N_, d_all_pairs_idxs_.data, d_col_idxs_.data);
        gpuErrchk(cudaPeekAtLastError());

        // Update the frozen indices so that only indices that are also in the allpairs indices are considered frozen
        k_idxs_intersection<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N_, d_col_idxs_.data, d_free_idxs_.data, d_temp_idxs_.data);
        gpuErrchk(cudaPeekAtLastError());
        d_free_idx_ptr = d_temp_idxs_.data;
    }

    // Partition the column idxs to the column buffer to setup the interaction group
    gpuErrchk(cub::DevicePartition::If(
        d_temp_storage_buffer_.data,
        temp_storage_bytes_,
        d_free_idx_ptr,
        d_col_idxs_.data,
        d_num_selected_buffer_.data,
        N_,
        select_op,
        stream));

    // Free particles should be in the row idxs
    set_nonbonded_ixn_potential_idxs(
        ixn_group_->potential, num_col_idxs, num_row_idxs, d_col_idxs_.data, d_row_idxs_.data, stream);

    // Invert to get back to the free indices, which the integrator will use
    k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());

    if (!freeze_reference) {
        // If there are no frozen indices, don't attach any restraints
        if (num_col_idxs > 0) {
            k_construct_bonded_params<<<ceil_divide(num_col_idxs, tpb), tpb, 0, stream>>>(
                num_col_idxs,
                N_,
                reference_idx,
                k,
                0.0,
                radius,
                d_col_idxs_.data,
                d_restraint_pairs_.data,
                d_bond_params_.data);
            gpuErrchk(cudaPeekAtLastError());
        }

        bound_frozen_restraint_->set_params_device(3 * num_col_idxs, d_bond_params_.data, stream);
        frozen_restraint_->set_bonds_device(num_col_idxs, d_restraint_pairs_.data, stream);
    }
}

std::vector<std::shared_ptr<BoundPotential>> LocalMDPotentials::get_potentials() { return all_potentials_; }

unsigned int *LocalMDPotentials::get_free_idxs() { return d_free_idxs_.data; }

// reset_potentials resets the potentials passed in to the constructor to be in the original state. This is because
// they are passed by reference and so changes made to the potentials will persist otherwise beyond the scope of the local md.
void LocalMDPotentials::reset_potentials(cudaStream_t stream) {
    // Set back to the original indices
    set_nonbonded_potential_idxs(nonbonded_bp_->potential, num_allpairs_idxs_, d_all_pairs_idxs_.data, stream);
}

} // namespace timemachine
