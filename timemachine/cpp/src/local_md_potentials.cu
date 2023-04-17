#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "kernels/k_flat_bottom_bond.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_local_md.cuh"
#include "local_md_potentials.hpp"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <random>
#include <vector>

namespace timemachine {

// Struct to as a CUB < operation
struct LessThan {
    int compare;
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ explicit LessThan(int compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ bool operator()(const int &a) const { return (a < compare); }
};

LocalMDPotentials::LocalMDPotentials(const int N, const std::vector<std::shared_ptr<BoundPotential>> bps)
    : N_(N), temp_storage_bytes_(0), all_potentials_(bps), restraints_(N_ * 2), bond_params_(N_ * 3),
      probability_buffer_(round_up_even(N_)), d_free_idxs_(N_), d_row_idxs_(N_), d_col_idxs_(N_), p_num_selected_(1),
      num_selected_buffer_(1) {

    std::vector<std::shared_ptr<BoundPotential>> nonbonded_pots;
    get_nonbonded_all_pair_potentials(bps, nonbonded_pots);

    if (nonbonded_pots.size() > 1) {
        throw std::runtime_error("found multiple NonbondedAllPairs potentials");
    }
    if (nonbonded_pots.size() != 1) {
        throw std::runtime_error("unable to find a NonbondedAllPairs potential");
    }

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
    restraint_ = std::shared_ptr<FlatBottomBond<double>>(new FlatBottomBond<double>(default_bonds));
    // Construct a bound potential with 0 params
    bound_restraint_ = std::shared_ptr<BoundPotential>(new BoundPotential(restraint_, std::vector<int>({0}), nullptr));

    ixn_group_ =
        construct_ixn_group_potential(N_, nonbonded_bp_->potential, nonbonded_bp_->size(), nonbonded_bp_->d_p->data);

    // Add the restraint potential and ixn group potential
    all_potentials_.push_back(bound_restraint_);
    all_potentials_.push_back(ixn_group_);

    cub::DevicePartition::If(
        nullptr, temp_storage_bytes_, d_free_idxs_.data, d_row_idxs_.data, num_selected_buffer_.data, N_, LessThan(N_));
    // Allocate char as temp_storage_bytes_ is in raw bytes and the type doesn't matter in practice.
    // Equivalent to DeviceBuffer<int> buf(temp_storage_bytes_ / sizeof(int))
    d_temp_storage_buffer_.reset(new DeviceBuffer<char>(temp_storage_bytes_));

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
};

LocalMDPotentials::~LocalMDPotentials() { curandErrchk(curandDestroyGenerator(cr_rng_)); }

// setup_from_idxs takes a set of idxs, a temperature and a seed to determine the free particles. Fix the local_idxs to length
// one to ensure the same reference everytime, though the seed also handles the probabilities of selecting particles, and it is suggested
// to provide a new seed at each step.
void LocalMDPotentials::setup_from_idxs(
    double *d_x_t,
    double *d_box_t,
    const std::vector<int> &local_idxs,
    const double temperature,
    const int seed,
    const double radius,
    const double k,
    const cudaStream_t stream) {

    curandErrchk(curandSetStream(cr_rng_, stream));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
    // Reset the generator offset to ensure same values for the same seed are produced
    // Simply reseeding does NOT produce identical results
    curandErrchk(curandSetGeneratorOffset(cr_rng_, 0));

    // Set the array to all N, which indicates to ignore that idx
    k_initialize_array<unsigned int><<<ceil_divide(N_, warp_size), warp_size, 0, stream>>>(N_, d_free_idxs_.data, N_);
    gpuErrchk(cudaPeekAtLastError());

    // Generate values between (0, 1.0]
    curandErrchk(curandGenerateUniform(cr_rng_, probability_buffer_.data, round_up_even(N_)));

    std::mt19937 rng;
    rng.seed(seed);
    std::uniform_int_distribution<unsigned int> random_dist(0, local_idxs.size() - 1);

    unsigned int reference_idx = local_idxs[random_dist(rng)];

    const double kBT = BOLTZ * temperature;
    // Select all of the particles that will be free
    k_log_probability_selection<float><<<ceil_divide(N_, warp_size), warp_size, 0, stream>>>(
        N_, kBT, radius, k, reference_idx, d_x_t, d_box_t, probability_buffer_.data, d_free_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());

    this->_setup_free_idxs_given_reference_idx(reference_idx, radius, k, stream);
}

void LocalMDPotentials::_setup_free_idxs_given_reference_idx(
    const unsigned int reference_idx, const double radius, const double k, const cudaStream_t stream) {
    const int tpb = warp_size;

    LessThan select_op(N_);

    // Partition the free idxs into the row idxs
    gpuErrchk(cub::DevicePartition::If(
        d_temp_storage_buffer_->data,
        temp_storage_bytes_,
        d_free_idxs_.data,
        d_row_idxs_.data,
        num_selected_buffer_.data,
        N_,
        select_op,
        stream));

    gpuErrchk(cudaMemcpyAsync(
        p_num_selected_.data,
        num_selected_buffer_.data,
        1 * sizeof(*p_num_selected_.data),
        cudaMemcpyDeviceToHost,
        stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    const int num_row_idxs = p_num_selected_.data[0];
    const int num_col_idxs = N_ - num_row_idxs;

    if (num_row_idxs == 0) {
        throw std::runtime_error("LocalMDPotentials setup has no free particles selected");
    }

    // The reference particle will always be in the column idxs
    if (num_row_idxs == N_ - 1) {
        fprintf(stderr, "LocalMDPotentials setup has entire system selected\n");
    }

    k_construct_bonded_params<<<ceil_divide(num_row_idxs, tpb), tpb, 0, stream>>>(
        num_row_idxs, N_, reference_idx, k, 0.0, radius, d_row_idxs_.data, restraints_.data, bond_params_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Setup the flat bottom restraints
    bound_restraint_->set_params_device(std::vector<int>({num_row_idxs, 3}), bond_params_.data, stream);
    restraint_->set_bonds_device(num_row_idxs, restraints_.data, stream);

    // Set the nonbonded potential to compute forces of free particles
    set_nonbonded_potential_idxs(nonbonded_bp_->potential, num_row_idxs, d_row_idxs_.data, stream);

    // Invert to get column idxs
    k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());

    // Partition the column idxs to the column buffer to setup the interaction group
    gpuErrchk(cub::DevicePartition::If(
        d_temp_storage_buffer_->data,
        temp_storage_bytes_,
        d_free_idxs_.data,
        d_col_idxs_.data,
        num_selected_buffer_.data,
        N_,
        select_op,
        stream));

    // Free particles should be in the row idxs
    set_nonbonded_ixn_potential_idxs(
        ixn_group_->potential, num_col_idxs, num_row_idxs, d_col_idxs_.data, d_row_idxs_.data, stream);
}

std::vector<std::shared_ptr<BoundPotential>> LocalMDPotentials::get_potentials() { return all_potentials_; }

DeviceBuffer<unsigned int> *LocalMDPotentials::get_free_idxs() { return &d_row_idxs_; }

void LocalMDPotentials::reset(const cudaStream_t stream) {
    // Set the row idxs back to the identity.
    k_arange<<<ceil_divide(N_, warp_size), warp_size, 0, stream>>>(N_, d_row_idxs_.data);
    gpuErrchk(cudaPeekAtLastError());
    // Set back to the full system
    set_nonbonded_potential_idxs(nonbonded_bp_->potential, N_, d_row_idxs_.data, stream);
}

} // namespace timemachine
