#include "constants.hpp"
#include "context.hpp"

#include "fixed_point.hpp"
#include "flat_bottom_bond.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_flat_bottom_bond.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_local_md.cuh"
#include "kernels/kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "local_md_utils.hpp"
#include "math_utils.cuh"
#include "pinned_host_buffer.hpp"
#include "set_utils.hpp"
#include <cub/cub.cuh>
#include <memory>
#include <random>

namespace timemachine {

Context::Context(
    int N,
    const double *x_0,
    const double *v_0,
    const double *box_0,
    std::shared_ptr<Integrator> intg,
    std::vector<std::shared_ptr<BoundPotential>> bps,
    std::shared_ptr<MonteCarloBarostat> barostat)
    : N_(N), barostat_(barostat), step_(0), d_sum_storage_(nullptr), d_sum_storage_bytes_(0), intg_(intg), bps_(bps) {
    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N * 3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N * 3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3 * 3);
    cudaSafeMalloc(&d_u_buffer_, N * sizeof(*d_u_buffer_));

    unsigned long long *d_in_tmp = nullptr;  // dummy
    unsigned long long *d_out_tmp = nullptr; // dummy

    // Compute the storage size necessary to reduce energies
    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_in_tmp, d_out_tmp, N_);
    gpuErrchk(cudaPeekAtLastError());
    cudaSafeMalloc(&d_sum_storage_, d_sum_storage_bytes_);
};

Context::~Context() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_box_t_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_storage_));
};

double Context::_get_temperature() {
    if (std::shared_ptr<LangevinIntegrator> langevin = std::dynamic_pointer_cast<LangevinIntegrator>(intg_);
        langevin != nullptr) {
        return langevin->get_temperature();
    } else {
        throw std::runtime_error("integrator must be LangevinIntegrator.");
    }
}

std::array<std::vector<double>, 2> Context::multiple_steps_local(
    const int n_steps,
    const std::vector<int> &local_idxs,
    const int burn_in,
    const int store_x_interval,
    const double radius,
    const double k,
    const int seed) {
    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }
    const double temperature = this->_get_temperature();

    const int x_buffer_size = n_steps / store_x_interval;

    const int box_buffer_size = x_buffer_size * 3 * 3;

    std::vector<std::shared_ptr<BoundPotential>> nonbonded_pots;
    get_nonbonded_all_pair_potentials(bps_, nonbonded_pots);

    if (nonbonded_pots.size() > 1) {
        throw std::runtime_error("found multiple NonbondedAllPairs potentials");
    }
    if (nonbonded_pots.size() != 1) {
        throw std::runtime_error("unable to find a NonbondedAllPairs potential");
    }

    // Only used to reference shared_ptr to potential and for Nonbonded parameters
    // modifications to the BoundPotential has no impact
    const auto nonbonded_bp = nonbonded_pots[0];

    std::shared_ptr<BoundPotential> ixn_group =
        construct_ixn_group_potential(N_, nonbonded_bp->potential, nonbonded_bp->size(), nonbonded_bp->d_p->data);

    std::mt19937 rng;
    rng.seed(seed);
    std::uniform_int_distribution<unsigned int> random_dist(0, local_idxs.size() - 1);

    // Store coordinates in host memory as it can be very large
    std::vector<double> h_x_buffer(x_buffer_size * N_ * 3);
    // Store boxes on GPU as boxes are a constant size and relatively small
    std::unique_ptr<DeviceBuffer<double>> d_box_traj(nullptr);
    if (box_buffer_size > 0) {
        d_box_traj.reset(new DeviceBuffer<double>(box_buffer_size));
    }

    const size_t tpb = warp_size;

    DeviceBuffer<unsigned int> d_free_indices(N_);

    DeviceBuffer<unsigned int> d_row_idxs(N_);
    DeviceBuffer<unsigned int> d_col_idxs(N_);

    // Pinned memory for getting lengths of indice arrays
    PinnedHostBuffer<int> p_num_selected(1);
    DeviceBuffer<int> num_selected_buffer(1);
    LessThan select_op(N_);

    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::If(
        nullptr, temp_storage_bytes, d_free_indices.data, d_row_idxs.data, num_selected_buffer.data, N_, select_op);
    // Allocate char as temp_storage_bytes is in raw bytes and the type doesn't matter in practice.
    // Equivalent to DeviceBuffer<int> buf(temp_storage_bytes / sizeof(int))
    DeviceBuffer<char> d_temp_storage_buffer(temp_storage_bytes);

    DeviceBuffer<int> restraints(N_ * 2);
    DeviceBuffer<double> bond_params(N_ * 3);
    // Ensure that we allocate enough space for all potential bonds
    std::vector<int> default_bonds(2 * N_);
    for (int i = 0; i < N_; i++) {
        default_bonds[i * 2 + 0] = 0;
        default_bonds[i * 2 + 1] = i + 1;
    }
    std::shared_ptr<FlatBottomBond<double>> restraint_ptr(new FlatBottomBond<double>(default_bonds));
    // Construct a bound potential with 0 params
    std::shared_ptr<BoundPotential> bound_shell_restraint(
        new BoundPotential(restraint_ptr, std::vector<int>({0}), nullptr));

    // Copy constructor to get copies of the BoundPotentials
    std::vector<std::shared_ptr<BoundPotential>> local_bps = bps_;
    // Add the restraint potential aand ixn group potential
    local_bps.push_back(bound_shell_restraint);
    local_bps.push_back(ixn_group);

    const double kBT = BOLTZ * temperature;

    cudaStream_t stream;

    curandGenerator_t cr_rng;
    DeviceBuffer<float> probability_buffer(round_up_even(N_));
    curandErrchk(curandCreateGenerator(&cr_rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng, seed));

    // Create stream that doesn't sync with the default stream
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    try {
        curandErrchk(curandSetStream(cr_rng, stream));

        // Set the array to all N, which indicates to ignore that idx
        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_indices.data, N_);
        gpuErrchk(cudaPeekAtLastError());
        // Generate values between (0, 1.0]
        curandErrchk(curandGenerateUniform(cr_rng, probability_buffer.data, round_up_even(N_)));

        unsigned int reference_idx = local_idxs[random_dist(rng)];

        // Select all of the particles that will be free
        k_log_probability_selection<float><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N_, kBT, radius, k, reference_idx, d_x_t_, d_box_t_, probability_buffer.data, d_free_indices.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition the free indices into the row indices
        gpuErrchk(cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_free_indices.data,
            d_row_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream));

        gpuErrchk(cudaMemcpyAsync(
            p_num_selected.data,
            num_selected_buffer.data,
            1 * sizeof(*p_num_selected.data),
            cudaMemcpyDeviceToHost,
            stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        int num_row_indices = p_num_selected.data[0];
        int num_col_indices = N_ - num_row_indices;

        if (num_row_indices == 0) {
            throw std::runtime_error("Context::multiple_steps_local(): no free particles selected");
        }

        // The reference particle will always be in the column indices
        if (num_row_indices == N_ - 1) {
            fprintf(stderr, "Context::multiple_steps_local(): entire system selected\n");
        }

        k_construct_bonded_params<<<ceil_divide(num_row_indices, tpb), tpb, 0, stream>>>(
            num_row_indices, N_, reference_idx, k, 0.0, radius, d_row_idxs.data, restraints.data, bond_params.data);
        gpuErrchk(cudaPeekAtLastError());
        // Setup the flat bottom restraints
        bound_shell_restraint->set_params_device(std::vector<int>({num_row_indices, 3}), bond_params.data, stream);
        restraint_ptr->set_bonds_device(num_row_indices, restraints.data, stream);

        // Set the nonbonded potential to compute forces of free particles
        set_nonbonded_potential_idxs(nonbonded_bp->potential, num_row_indices, d_row_idxs.data, stream);

        // Invert to get column indices
        k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_free_indices.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition the column indices to the column buffer to setup the interaction group
        gpuErrchk(cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_free_indices.data,
            d_col_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream));

        // Free particles should be in the row indices
        set_nonbonded_ixn_potential_idxs(
            ixn_group->potential, num_col_indices, num_row_indices, d_col_idxs.data, d_row_idxs.data, stream);

        intg_->initialize(local_bps, d_x_t_, d_v_t_, d_box_t_, d_row_idxs.data, stream);
        for (int i = 0; i < burn_in; i++) {
            this->_step(local_bps, d_row_idxs.data, stream);
        }
        for (int i = 1; i <= n_steps; i++) {
            this->_step(local_bps, d_row_idxs.data, stream);
            if (i % store_x_interval == 0) {
                gpuErrchk(cudaMemcpyAsync(
                    &h_x_buffer[0] + ((i / store_x_interval) - 1) * N_ * 3,
                    d_x_t_,
                    N_ * 3 * sizeof(double),
                    cudaMemcpyDeviceToHost,
                    stream));
                gpuErrchk(cudaMemcpyAsync(
                    &d_box_traj->data[0] + ((i / store_x_interval) - 1) * 3 * 3,
                    d_box_t_,
                    3 * 3 * sizeof(*d_box_traj->data),
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
        intg_->finalize(local_bps, d_x_t_, d_v_t_, d_box_t_, d_row_idxs.data, stream);

        // Set the row indices back to the identity.
        k_arange<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_row_idxs.data);
        gpuErrchk(cudaPeekAtLastError());
        // Set back to the full system, for when the loop ends
        set_nonbonded_potential_idxs(nonbonded_bp->potential, N_, d_row_idxs.data, stream);
    } catch (...) {
        gpuErrchk(cudaStreamSynchronize(stream));
        gpuErrchk(cudaStreamDestroy(stream));
        curandErrchk(curandDestroyGenerator(cr_rng));
        throw;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));
    curandErrchk(curandDestroyGenerator(cr_rng));

    std::vector<double> h_box_buffer(box_buffer_size);

    if (box_buffer_size > 0) {
        d_box_traj->copy_to(&h_box_buffer[0]);
    }
    return std::array<std::vector<double>, 2>({h_x_buffer, h_box_buffer});
}

std::array<std::vector<double>, 2> Context::multiple_steps(const int n_steps, int store_x_interval) {
    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }
    if (n_steps % store_x_interval != 0) {
        std::cout << "warning:: n_steps modulo store_x_interval does not equal zero" << std::endl;
    }

    int x_buffer_size = n_steps / store_x_interval;
    int box_buffer_size = x_buffer_size * 3 * 3;

    std::vector<double> h_x_buffer(x_buffer_size * N_ * 3);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    std::unique_ptr<DeviceBuffer<double>> d_box_buffer(nullptr);
    if (box_buffer_size > 0) {
        d_box_buffer.reset(new DeviceBuffer<double>(box_buffer_size));
    }

    intg_->initialize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    for (int i = 1; i <= n_steps; i++) {
        this->_step(bps_, nullptr, stream);

        if (i % store_x_interval == 0) {
            gpuErrchk(cudaMemcpyAsync(
                &h_x_buffer[0] + ((i / store_x_interval) - 1) * N_ * 3,
                d_x_t_,
                N_ * 3 * sizeof(double),
                cudaMemcpyDeviceToHost,
                stream));
            gpuErrchk(cudaMemcpyAsync(
                &d_box_buffer->data[0] + ((i / store_x_interval) - 1) * 3 * 3,
                d_box_t_,
                3 * 3 * sizeof(*d_box_buffer->data),
                cudaMemcpyDeviceToDevice,
                stream));
        }
    }
    intg_->finalize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);

    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<double> h_box_buffer(box_buffer_size);
    if (box_buffer_size > 0) {
        d_box_buffer->copy_to(&h_box_buffer[0]);
    }

    return std::array<std::vector<double>, 2>({h_x_buffer, h_box_buffer});
}

std::array<std::vector<double>, 3>
Context::multiple_steps_U(const int n_steps, int store_u_interval, int store_x_interval) {

    if (store_u_interval <= 0) {
        throw std::runtime_error("store_u_interval <= 0");
    }

    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }

    if (n_steps % store_x_interval != 0) {
        std::cout << "warning:: n_steps modulo store_x_interval does not equal zero" << std::endl;
    }

    if (n_steps % store_u_interval != 0) {
        std::cout << "warning:: n_steps modulo store_u_interval does not equal zero" << std::endl;
    }

    int u_traj_size = n_steps / store_u_interval;
    int x_traj_size = n_steps / store_x_interval;

    std::vector<double> h_x_traj(x_traj_size * N_ * 3);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    std::unique_ptr<DeviceBuffer<double>> d_box_traj(nullptr);
    if (x_traj_size > 0) {
        d_box_traj.reset(new DeviceBuffer<double>(x_traj_size * 3 * 3));
    }
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_u_traj(nullptr);
    if (u_traj_size > 0) {
        d_u_traj.reset(new DeviceBuffer<unsigned long long>(u_traj_size));
        gpuErrchk(cudaMemsetAsync(d_u_traj->data, 0, d_u_traj->size, stream));
    }

    intg_->initialize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    for (int step = 1; step <= n_steps; step++) {

        this->_step(bps_, nullptr, stream);

        if (step % store_x_interval == 0) {
            gpuErrchk(cudaMemcpyAsync(
                &h_x_traj[0] + ((step / store_x_interval) - 1) * N_ * 3,
                d_x_t_,
                N_ * 3 * sizeof(double),
                cudaMemcpyDeviceToHost,
                stream));
            gpuErrchk(cudaMemcpyAsync(
                &d_box_traj->data[0] + ((step / store_x_interval) - 1) * 3 * 3,
                d_box_t_,
                3 * 3 * sizeof(*d_box_traj->data),
                cudaMemcpyDeviceToDevice,
                stream));
        }

        // we need to compute aggregate energies
        if (u_traj_size > 0 && step % store_u_interval == 0) {
            // reset buffers on each pass.
            gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N_ * sizeof(*d_u_buffer_), stream));
            unsigned long long *u_ptr = d_u_traj->data + (step / store_u_interval) - 1;
            for (int i = 0; i < bps_.size(); i++) {
                bps_[i]->execute_device(N_, d_x_t_, d_box_t_, nullptr, nullptr, d_u_buffer_, stream);
            }
            cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_u_buffer_, u_ptr, N_, stream);
            gpuErrchk(cudaPeekAtLastError());
        }
    }
    intg_->finalize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);

    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<unsigned long long> h_u_traj_ull(u_traj_size);
    if (u_traj_size > 0) {
        d_u_traj->copy_to(&h_u_traj_ull[0]);
    }

    std::vector<double> h_u_traj_double(u_traj_size);
    for (int i = 0; i < h_u_traj_ull.size(); i++) {
        h_u_traj_double[i] = FIXED_TO_FLOAT<double>(h_u_traj_ull[i]);
    }
    std::vector<double> h_box_traj(x_traj_size * 3 * 3);
    if (x_traj_size > 0) {
        d_box_traj->copy_to(&h_box_traj[0]);
    }

    return std::array<std::vector<double>, 3>({h_u_traj_double, h_x_traj, h_box_traj});
}

void Context::step() {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->_step(bps_, nullptr, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

void Context::finalize() {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    intg_->finalize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

void Context::initialize() {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    intg_->initialize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

void Context::_step(
    std::vector<std::shared_ptr<BoundPotential>> &bps, unsigned int *d_atom_idxs, const cudaStream_t stream) {
    intg_->step_fwd(bps, d_x_t_, d_v_t_, d_box_t_, d_atom_idxs, stream);

    // If atom idxs are passed, indicates that only a subset of the system should move. Don't
    // run the barostat in this situation.
    if (d_atom_idxs == nullptr && barostat_) {
        // May modify coords, du_dx and box size
        barostat_->inplace_move(d_x_t_, d_box_t_, stream);
    }

    step_ += 1;
};

int Context::num_atoms() const { return N_; }

void Context::set_x_t(const double *in_buffer) {
    gpuErrchk(cudaMemcpy(d_x_t_, in_buffer, N_ * 3 * sizeof(*in_buffer), cudaMemcpyHostToDevice));
}

void Context::set_v_t(const double *in_buffer) {
    gpuErrchk(cudaMemcpy(d_v_t_, in_buffer, N_ * 3 * sizeof(*in_buffer), cudaMemcpyHostToDevice));
}

void Context::get_x_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_x_t_, N_ * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_v_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_v_t_, N_ * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_box(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_box_t_, 3 * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

} // namespace timemachine
