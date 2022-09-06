#include "context.hpp"
#include "fanout_summed_potential.hpp"
#include "fixed_point.hpp"
#include "flat_bottom_bond.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_flat_bottom_bond.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_local_md.cuh"
#include "kernels/kernel_utils.cuh"
#include "neighborlist.hpp"
#include "nonbonded_all_pairs.hpp"
#include "set_utils.hpp"
#include "summed_potential.hpp"
#include "thermostat.hpp"
#include <cub/cub.cuh>
#include <memory>
#include <random>
#include <typeinfo>

#define BOLTZ 0.008314462618

namespace timemachine {

Context::Context(
    int N,
    const double *x_0,
    const double *v_0,
    const double *box_0,
    Integrator *intg,
    std::vector<BoundPotential *> bps,
    MonteCarloBarostat *barostat)
    : N_(N), barostat_(barostat), step_(0), d_sum_storage_(nullptr), d_sum_storage_bytes_(0), intg_(intg), bps_(bps) {
    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N * 3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N * 3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3 * 3);
    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, N * sizeof(*d_du_dl_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, N * sizeof(*d_u_buffer_)));

    unsigned long long *d_in_tmp = nullptr;  // dummy
    unsigned long long *d_out_tmp = nullptr; // dummy

    // Compute the storage size necessary to reduce du_dl
    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_in_tmp, d_out_tmp, N_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sum_storage_, d_sum_storage_bytes_));

    // for(int i=0; i < bps.size(); i++) {
    // cudaStream_t stream;
    // gpuErrchk(cudaStreamCreate(&stream));
    // streams_.push_back(stream);
    // }
};

Context::~Context() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_box_t_));
    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_storage_));

    // for(int i=0; i < streams_.size(); i++) {
    // gpuErrchk(cudaStreamDestroy(streams_[i]));
    // }
};

double get_nonbonded_potential_cutoff(std::shared_ptr<Potential> pot) {
    if (std::shared_ptr<NonbondedAllPairs<float, true>> nb_pot =
            std::dynamic_pointer_cast<NonbondedAllPairs<float, true>>(pot);
        nb_pot) {
        return nb_pot->get_cutoff();
    } else if (std::shared_ptr<NonbondedAllPairs<float, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<float, false>>(pot);
               nb_pot) {
        return nb_pot->get_cutoff();
    } else if (std::shared_ptr<NonbondedAllPairs<double, true>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, true>>(pot);
               nb_pot) {
        return nb_pot->get_cutoff();
    } else if (std::shared_ptr<NonbondedAllPairs<double, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, false>>(pot);
               nb_pot) {
        return nb_pot->get_cutoff();
    } else {
        throw std::runtime_error("Unable to cast potential to NonbondedAllPairs");
    }
}

bool is_nonbonded_potential(std::shared_ptr<Potential> pot) {
    if (std::shared_ptr<NonbondedAllPairs<float, true>> nb_pot =
            std::dynamic_pointer_cast<NonbondedAllPairs<float, true>>(pot);
        nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<float, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<float, false>>(pot);
               nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<double, true>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, true>>(pot);
               nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<double, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, false>>(pot);
               nb_pot) {
        return true;
    }
    return false;
}

void set_nonbonded_potential_idxs(
    std::shared_ptr<Potential> pot, const int num_idxs, const unsigned int *d_idxs, const cudaStream_t stream) {
    if (std::shared_ptr<NonbondedAllPairs<float, true>> nb_pot =
            std::dynamic_pointer_cast<NonbondedAllPairs<float, true>>(pot);
        nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<float, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<float, false>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<double, true>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, true>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<double, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, false>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
    }
}

void Context::_assert_temperature_matches(const double temperature) {
    if (Thermostat *thermostat = dynamic_cast<Thermostat *>(intg_); thermostat != nullptr) {
        if (thermostat->get_temperature() != temperature) {
            throw std::runtime_error("Local MD temperature didn't match Thermostat's temperature.");
        }
    }
}

// Recursively flatten the potentials. Important to find specific NonbondedAllPairs potentials for multiple_steps_local which
// can be wrapped in FanoutSummedPotential or SummedPotential objects.
void flatten_potentials(
    std::vector<std::shared_ptr<Potential>> input, std::vector<std::shared_ptr<Potential>> &flattened) {
    for (std::shared_ptr<Potential> pot : input) {
        std::shared_ptr<FanoutSummedPotential> fanned_potential = std::dynamic_pointer_cast<FanoutSummedPotential>(pot);
        if (fanned_potential != nullptr) {
            flatten_potentials(fanned_potential->get_potentials(), flattened);
            continue;
        }
        std::shared_ptr<SummedPotential> summed_potential = std::dynamic_pointer_cast<SummedPotential>(pot);
        if (summed_potential != nullptr) {
            flatten_potentials(summed_potential->get_potentials(), flattened);
            continue;
        }
        flattened.push_back(pot);
    }
}

std::array<std::vector<double>, 2> Context::multiple_steps_local(
    const std::vector<double> &lambda_schedule,
    const std::vector<unsigned int> &local_idxs,
    const int store_x_interval,
    const double radius,
    const double k,
    const double temperature,
    const int seed) {
    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }
    this->_assert_temperature_matches(temperature);
    const int x_buffer_size = lambda_schedule.size() / store_x_interval;

    const int box_buffer_size = x_buffer_size * 3 * 3;

    std::vector<std::shared_ptr<Potential>> initial_potentials(bps_.size());

    for (BoundPotential *pot : bps_) {
        initial_potentials.push_back(pot->potential);
    }
    std::vector<std::shared_ptr<Potential>> flattened_potentials;
    flatten_potentials(initial_potentials, flattened_potentials);

    std::shared_ptr<Potential> nonbonded_potential;

    // Find the nonbonded potential
    for (std::shared_ptr<Potential> pot : flattened_potentials) {
        if (is_nonbonded_potential(pot)) {
            if (nonbonded_potential) {
                throw std::runtime_error("found multiple NonbondedAllPairs potentials");
            }
            nonbonded_potential = pot;
        }
    }
    if (!nonbonded_potential) {
        throw std::runtime_error("unable to find a NonbondedAllPairs potential");
    }
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    curandGenerator_t cr_rng;
    DeviceBuffer<float> probability_buffer(round_up_even(N_));
    curandErrchk(curandCreateGenerator(&cr_rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng, seed));
    curandErrchk(curandSetStream(cr_rng, stream));

    std::mt19937 rng;
    rng.seed(seed);
    std::uniform_int_distribution<unsigned int> random_dist(0, local_idxs.size() - 1);

    // Construct neighborlist to find the inner and outer shell
    Neighborlist<float> nblist(N_);

    // Store coordinates in host memory as it can be very large
    std::vector<double> h_x_buffer(x_buffer_size * N_ * 3);
    // Store boxes on GPU as boxes are a constant size and relatively small1
    DeviceBuffer<double> d_box_buffer(box_buffer_size);

    const size_t tpb = warp_size;

    DeviceBuffer<unsigned int> d_shell_idxs_inner(N_);

    DeviceBuffer<unsigned int> d_row_idxs(N_);
    // d_col indices used both for column indices for neighborlist as well as outer shell.
    DeviceBuffer<unsigned int> d_col_idxs(N_);

    // Pinned memory for getting lengths of indice arrays
    DeviceBuffer<int, true> p_num_selected(1);
    DeviceBuffer<int> num_selected_buffer(1);
    LessThan select_op(N_);

    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::If(
        nullptr, temp_storage_bytes, d_shell_idxs_inner.data, d_row_idxs.data, num_selected_buffer.data, N_, select_op);
    DeviceBuffer<char> d_temp_storage_buffer(temp_storage_bytes);

    const double outer_cutoff = get_nonbonded_potential_cutoff(nonbonded_potential);

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
    BoundPotential bound_shell_restraint(restraint_ptr, std::vector<int>({0}), nullptr);

    // Copy constructor to get new set of bound potentials
    std::vector<BoundPotential *> local_bps = bps_;
    local_bps.push_back(&bound_shell_restraint);

    const double kBt = BOLTZ * temperature;

    try {

        // Set the array to all N, which means it will be ignored as an idx
        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_shell_idxs_inner.data, N_);
        gpuErrchk(cudaPeekAtLastError());
        // Generate values between (0, 1.0]
        curandErrchk(curandGenerateUniform(cr_rng, probability_buffer.data, round_up_even(N_)));

        unsigned int reference_idx = local_idxs[random_dist(rng)];

        k_log_probability_selection<double><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
            N_, kBt, radius, k, reference_idx, d_x_t_, d_box_t_, probability_buffer.data, d_shell_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition the valid row indices to the front of the array, defines the complete inner shell
        cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_shell_idxs_inner.data,
            d_row_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream);
        gpuErrchk(cudaPeekAtLastError());

        // Copy the num out, that is the new num_row_indices, num_col_indices == N_ - num_row_indices
        gpuErrchk(cudaMemcpyAsync(
            p_num_selected.data,
            num_selected_buffer.data,
            1 * sizeof(*p_num_selected.data),
            cudaMemcpyDeviceToHost,
            stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        int num_row_indices = p_num_selected.data[0];
        int num_col_indices = N_ - num_row_indices;
        if (num_row_indices == 0 || num_col_indices == 0) {
            throw std::runtime_error("local md no longer stable, check system");
        }

        k_construct_bonded_params<<<ceil_divide(num_row_indices, tpb), tpb, 0, stream>>>(
            num_row_indices, N_, reference_idx, k, 0.0, radius, d_row_idxs.data, restraints.data, bond_params.data);
        gpuErrchk(cudaPeekAtLastError());
        // Hacky
        bound_shell_restraint.set_params_device(std::vector<int>({num_row_indices, 3}), bond_params.data, stream);
        restraint_ptr->set_bonds_device(num_row_indices, restraints.data, stream);
        // Invert to get the column indices
        k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_shell_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition the col idxs to the front
        cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_shell_idxs_inner.data,
            d_col_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream);
        gpuErrchk(cudaPeekAtLastError());

        // Invert to get back to the inner shell idxs
        k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_shell_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        nblist.set_idxs_device(num_col_indices, num_row_indices, d_col_idxs.data, d_row_idxs.data, stream);
        int max_interactions = nblist.max_ixn_count();
        // Build the neighborlist around the inner idxs to get the outer shell. Use the nonbonded potential's cutoff
        // to ensure correctness and to avoid wasted computation (ie radius >> outer_cutoff).
        nblist.build_nblist_device(N_, d_x_t_, d_box_t_, outer_cutoff, stream);

        // Now reuse the d_col_idxs for the outer idxs, to reduce memory consumption
        // Set the array to all N, which means it will be ignored as an idx
        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_col_idxs.data, N_);
        gpuErrchk(cudaPeekAtLastError());

        k_unique_indices<<<ceil_divide(max_interactions, tpb), tpb, 0, stream>>>(
            max_interactions, N_, nblist.get_ixn_atoms(), d_col_idxs.data);
        gpuErrchk(cudaPeekAtLastError());

        // Add the inner indices to the outer indices
        k_unique_indices<<<ceil_divide(num_row_indices, tpb), tpb, 0, stream>>>(
            num_row_indices, N_, d_row_idxs.data, d_col_idxs.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition all the indices that make up the the inner and outer shell, reuse the d_row_idxs
        cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_col_idxs.data,
            d_row_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream);
        gpuErrchk(cudaPeekAtLastError());

        // Copy out the number of indices in the outer indices to the row indices
        gpuErrchk(cudaMemcpyAsync(
            p_num_selected.data,
            num_selected_buffer.data,
            1 * sizeof(*p_num_selected.data),
            cudaMemcpyDeviceToHost,
            stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        // Set the nonbonded potential to compute forces of inner+outer shell.
        set_nonbonded_potential_idxs(nonbonded_potential, p_num_selected.data[0], d_row_idxs.data, stream);
        for (int i = 1; i <= lambda_schedule.size(); i++) {
            this->_step(local_bps, lambda_schedule[i - 1], nullptr, d_shell_idxs_inner.data, stream);
            if (i % store_x_interval == 0) {
                gpuErrchk(cudaMemcpyAsync(
                    d_box_buffer.data + ((i / store_x_interval) - 1) * 3 * 3,
                    d_box_t_,
                    3 * 3 * sizeof(*d_box_buffer.data),
                    cudaMemcpyDeviceToDevice,
                    stream));
                gpuErrchk(cudaMemcpy(
                    &h_x_buffer[0] + ((i / store_x_interval) - 1) * N_ * 3,
                    d_x_t_,
                    N_ * 3 * sizeof(*d_x_t_),
                    cudaMemcpyDeviceToHost));
            }
        }
        // Set the row indices back to the identity.
        k_arange<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_row_idxs.data);
        gpuErrchk(cudaPeekAtLastError());
        // Set back to the full system, for when the loop ends
        set_nonbonded_potential_idxs(nonbonded_potential, N_, d_row_idxs.data, stream);
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
    d_box_buffer.copy_to(&h_box_buffer[0]);
    return std::array<std::vector<double>, 2>({h_x_buffer, h_box_buffer});
}

std::array<std::vector<double>, 3>
Context::multiple_steps(const std::vector<double> &lambda_schedule, int store_du_dl_interval, int store_x_interval) {
    unsigned long long *d_du_dl_buffer = nullptr;
    double *d_box_buffer = nullptr;
    // try catch block is to deal with leaks in d_du_dl_buffer
    if (store_du_dl_interval <= 0) {
        throw std::runtime_error("store_du_dl_interval <= 0");
    }
    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }
    if (lambda_schedule.size() % store_x_interval != 0) {
        std::cout << "warning:: length of lambda_schedule modulo store_x_interval does not equal zero" << std::endl;
    }

    if (lambda_schedule.size() % store_du_dl_interval != 0) {
        std::cout << "warning:: length of lambda_schedule modulo store_du_dl_interval does not equal zero" << std::endl;
    }

    int du_dl_buffer_size = lambda_schedule.size() / store_du_dl_interval;
    int x_buffer_size = lambda_schedule.size() / store_x_interval;
    int box_buffer_size = x_buffer_size * 3 * 3;

    std::vector<double> h_x_buffer(x_buffer_size * N_ * 3);
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    try {
        gpuErrchk(cudaMalloc(&d_box_buffer, box_buffer_size * sizeof(*d_box_buffer)));
        // indicator so we can set it to a default arg.
        gpuErrchk(cudaMalloc(&d_du_dl_buffer, du_dl_buffer_size * sizeof(*d_du_dl_buffer)));
        gpuErrchk(cudaMemset(d_du_dl_buffer, 0, du_dl_buffer_size * sizeof(*d_du_dl_buffer)));
        intg_->initialize(bps_, lambda_schedule[0], d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
        for (int i = 1; i <= lambda_schedule.size(); i++) {
            // decide if we need to store the du_dl for this step
            unsigned long long *du_dl_ptr = nullptr;
            if (i % store_du_dl_interval == 0) {
                // pemdas but just to make it clear we're doing pointer arithmetic
                du_dl_ptr = d_du_dl_buffer + ((i / store_du_dl_interval) - 1);
            }

            double lambda = lambda_schedule[i - 1];
            this->_step(bps_, lambda, du_dl_ptr, nullptr, stream);

            if (i % store_x_interval == 0) {
                gpuErrchk(cudaMemcpy(
                    &h_x_buffer[0] + ((i / store_x_interval) - 1) * N_ * 3,
                    d_x_t_,
                    N_ * 3 * sizeof(double),
                    cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(
                    &d_box_buffer[0] + ((i / store_x_interval) - 1) * 3 * 3,
                    d_box_t_,
                    3 * 3 * sizeof(*d_box_buffer),
                    cudaMemcpyDeviceToDevice));
            }
        }
        intg_->finalize(bps_, lambda_schedule[lambda_schedule.size() - 1], d_x_t_, d_v_t_, d_box_t_, nullptr, stream);

        gpuErrchk(cudaDeviceSynchronize());

        std::vector<unsigned long long> h_du_dl_buffer_ull(du_dl_buffer_size);
        gpuErrchk(cudaMemcpy(
            &h_du_dl_buffer_ull[0],
            d_du_dl_buffer,
            du_dl_buffer_size * sizeof(*d_du_dl_buffer),
            cudaMemcpyDeviceToHost));

        std::vector<double> h_du_dl_buffer_double(du_dl_buffer_size);
        for (int i = 0; i < h_du_dl_buffer_ull.size(); i++) {
            h_du_dl_buffer_double[i] = FIXED_TO_FLOAT<double>(h_du_dl_buffer_ull[i]);
        }
        std::vector<double> h_box_buffer(box_buffer_size);
        gpuErrchk(cudaMemcpy(
            &h_box_buffer[0], d_box_buffer, box_buffer_size * sizeof(*d_box_buffer), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_du_dl_buffer));
        gpuErrchk(cudaFree(d_box_buffer));
        return std::array<std::vector<double>, 3>({h_du_dl_buffer_double, h_x_buffer, h_box_buffer});

    } catch (...) {
        gpuErrchk(cudaFree(d_du_dl_buffer));
        gpuErrchk(cudaFree(d_box_buffer));
        throw;
    }
}

std::array<std::vector<double>, 3> Context::multiple_steps_U(
    const double lambda, // which lambda window we run the integrator over
    const int n_steps,
    const std::vector<double> &lambda_windows, // which lambda windows we wish to evaluate U at
    int store_u_interval,
    int store_x_interval) {

    unsigned long long *d_u_traj = nullptr;
    double *d_box_traj = nullptr;

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

    int n_windows = lambda_windows.size();
    int u_traj_size = (n_steps / store_u_interval) * n_windows;
    int x_traj_size = n_steps / store_x_interval;
    int box_traj_size = x_traj_size * 3 * 3;

    std::vector<double> h_x_traj(x_traj_size * N_ * 3);

    // try catch block is to deal with leaks in d_u_buffer
    try {
        gpuErrchk(cudaMalloc(&d_box_traj, box_traj_size * sizeof(*d_box_traj)));
        gpuErrchk(cudaMalloc(&d_u_traj, u_traj_size * sizeof(*d_u_traj)));
        gpuErrchk(cudaMemset(d_u_traj, 0, u_traj_size * sizeof(*d_u_traj)));

        cudaStream_t stream = static_cast<cudaStream_t>(0);
        intg_->initialize(bps_, lambda, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
        for (int step = 1; step <= n_steps; step++) {

            this->_step(bps_, lambda, nullptr, nullptr, stream);

            if (step % store_x_interval == 0) {
                gpuErrchk(cudaMemcpy(
                    &h_x_traj[0] + ((step / store_x_interval) - 1) * N_ * 3,
                    d_x_t_,
                    N_ * 3 * sizeof(double),
                    cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(
                    &d_box_traj[0] + ((step / store_x_interval) - 1) * 3 * 3,
                    d_box_t_,
                    3 * 3 * sizeof(*d_box_traj),
                    cudaMemcpyDeviceToDevice));
            }

            // we need to compute aggregate energies
            if (step % store_u_interval == 0) {
                unsigned long long *u_ptr = d_u_traj + ((step / store_u_interval) - 1) * n_windows;
                for (int w = 0; w < n_windows; w++) {
                    // reset buffers on each pass.
                    gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N_ * sizeof(*d_u_buffer_), stream));
                    for (int i = 0; i < bps_.size(); i++) {
                        bps_[i]->execute_device(
                            N_, d_x_t_, d_box_t_, lambda_windows[w], nullptr, nullptr, nullptr, d_u_buffer_, stream);
                    }
                    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_u_buffer_, u_ptr + w, N_, stream);
                    gpuErrchk(cudaPeekAtLastError());
                }
            }
        }
        intg_->finalize(bps_, lambda, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);

        gpuErrchk(cudaDeviceSynchronize());

        std::vector<unsigned long long> h_u_traj_ull(u_traj_size);
        gpuErrchk(cudaMemcpy(&h_u_traj_ull[0], d_u_traj, u_traj_size * sizeof(*d_u_traj), cudaMemcpyDeviceToHost));

        std::vector<double> h_u_traj_double(u_traj_size);
        for (int i = 0; i < h_u_traj_ull.size(); i++) {
            h_u_traj_double[i] = FIXED_TO_FLOAT<double>(h_u_traj_ull[i]);
        }
        std::vector<double> h_box_traj(box_traj_size);
        gpuErrchk(cudaMemcpy(&h_box_traj[0], d_box_traj, box_traj_size * sizeof(*d_box_traj), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_u_traj));
        gpuErrchk(cudaFree(d_box_traj));
        return std::array<std::vector<double>, 3>({h_u_traj_double, h_x_traj, h_box_traj});

    } catch (...) {
        gpuErrchk(cudaFree(d_u_traj));
        gpuErrchk(cudaFree(d_box_traj));
        throw;
    }
}

void Context::step(double lambda) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->_step(bps_, lambda, nullptr, nullptr, stream);
    gpuErrchk(cudaDeviceSynchronize());
}

void Context::finalize(double lambda) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    intg_->finalize(bps_, lambda, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

void Context::initialize(double lambda) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    intg_->initialize(bps_, lambda, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

void Context::_step(
    std::vector<BoundPotential *> &bps,
    const double lambda,
    unsigned long long *du_dl_out,
    unsigned int *atom_idxs,
    const cudaStream_t stream) {

    if (du_dl_out) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, N_ * sizeof(*d_du_dl_buffer_), stream));
    }

    intg_->step_fwd(bps, lambda, d_x_t_, d_v_t_, d_box_t_, du_dl_out ? d_du_dl_buffer_ : nullptr, atom_idxs, stream);

    // compute du_dl
    if (du_dl_out) {
        cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_du_dl_buffer_, du_dl_out, N_, stream);
        gpuErrchk(cudaPeekAtLastError());
    }

    // If atom idxs are passed, indicates that only a subset of the system should move. Don't
    // run the barostat in this situation.
    if (atom_idxs == nullptr && barostat_) {
        // May modify coords, du_dx and box size
        barostat_->inplace_move(d_x_t_, d_box_t_, lambda, stream);
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
