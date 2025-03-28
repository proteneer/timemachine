#include "barostat.hpp"
#include "bound_potential.hpp"
#include "constants.hpp"
#include "context.hpp"

#include "fixed_point.hpp"
#include "flat_bottom_bond.hpp"
#include "gpu_utils.cuh"
#include "integrator.hpp"
#include "kernels/kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "local_md_potentials.hpp"
#include "math_utils.cuh"
#include "nonbonded_common.hpp"
#include "pinned_host_buffer.hpp"
#include "set_utils.hpp"

namespace timemachine {

static bool is_barostat(std::shared_ptr<Mover> &mover) {
    if (std::shared_ptr<MonteCarloBarostat<float>> baro = std::dynamic_pointer_cast<MonteCarloBarostat<float>>(mover);
        baro) {
        return true;
    }
    return false;
}

Context::Context(
    int N,
    const double *x_0,
    const double *v_0,
    const double *box_0,
    std::shared_ptr<Integrator> intg,
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    std::vector<std::shared_ptr<Mover>> &movers)
    : N_(N), movers_(movers), step_(0), intg_(intg), bps_(bps), nonbonded_pots_(0) {

    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N * 3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N * 3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3 * 3);

    // A no-op if running in vacuum or there are no NonbondedAllPairs potentials
    get_nonbonded_all_pair_potentials(bps, nonbonded_pots_);
};

Context::~Context() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_box_t_));
};

void Context::_verify_coords_and_box(const double *coords_buffer, const double *box_buffer, cudaStream_t stream) {
    // If there are no nonbonded potentials (ie Vacuum), nothing to check.
    if (nonbonded_pots_.size() == 0) {
        return;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    for (auto boundpot : nonbonded_pots_) {
        double cutoff = get_nonbonded_all_pair_cutoff_with_padding(boundpot->potential);
        double db_cutoff = 2 * cutoff;
        for (int i = 0; i < 3; i++) {
            if (box_buffer[i * 3 + i] < db_cutoff) {
                throw std::runtime_error(
                    "cutoff with padding is more than half of the box width, neighborlist is no longer reliable");
            }
        }
    }

    const double max_box_dim = max(box_buffer[0 * 3 + 0], max(box_buffer[1 * 3 + 1], box_buffer[2 * 3 + 2]));
    const auto [min_coord, max_coord] = std::minmax_element(coords_buffer, coords_buffer + N_ * 3);
    // Look at the largest difference in all dimensions, since coordinates are not imaged into the home box
    // per se, rather into the nearest periodic box
    const double max_coord_delta = *max_coord - *min_coord;
    if (max_box_dim * 100.0 < max_coord_delta) {
        throw std::runtime_error(
            "simulation unstable: dimensions of coordinates two orders of magnitude larger than max box dimension");
    }
}

double Context::_get_temperature() {
    if (std::shared_ptr<LangevinIntegrator<float>> langevin =
            std::dynamic_pointer_cast<LangevinIntegrator<float>>(intg_);
        langevin != nullptr) {
        return langevin->get_temperature();
    } else {
        throw std::runtime_error("integrator must be LangevinIntegrator.");
    }
}

void Context::setup_local_md(double temperature, bool freeze_reference, double ixn_group_nblist_padding) {
    if (this->local_md_pots_ != nullptr) {
        if (this->local_md_pots_->temperature != temperature ||
            this->local_md_pots_->freeze_reference != freeze_reference ||
            this->local_md_pots_->ixn_group_nblist_padding != ixn_group_nblist_padding) {
            throw std::runtime_error(
                "local md configured with different parameters, current parameters: Temperature " +
                std::to_string(this->local_md_pots_->temperature) + " Freeze Reference " +
                std::to_string(this->local_md_pots_->freeze_reference) + " Interaction Group Neighborlist padding " +
                std::to_string(this->local_md_pots_->ixn_group_nblist_padding));
        }
        return;
    }
    this->local_md_pots_.reset(
        new LocalMDPotentials(N_, bps_, freeze_reference, temperature, ixn_group_nblist_padding));
}

void Context::_ensure_local_md_intialized() {
    if (this->local_md_pots_ == nullptr) {
        double temperature = this->_get_temperature();
        this->setup_local_md(temperature, true);
    }
}

void Context::multiple_steps_local(
    const int n_steps,
    const std::vector<int> &local_idxs,
    const int n_samples,
    const double radius,
    const double k,
    const int seed,
    double *h_x,
    double *h_box) {
    const int store_x_interval = n_samples > 0 ? n_steps / n_samples : n_steps + 1;
    if (n_samples < 0) {
        throw std::runtime_error("n_samples < 0");
    }
    if (n_steps % store_x_interval != 0) {
        std::cout << "warning:: n_steps modulo store_x_interval does not equal zero" << std::endl;
    }
    this->_ensure_local_md_intialized();

    cudaStream_t stream;

    // Create stream that doesn't sync with the default stream
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {

        local_md_pots_->setup_from_idxs(d_x_t_, d_box_t_, local_idxs, seed, radius, k, stream);

        unsigned int *d_free_idxs = local_md_pots_->get_free_idxs();

        std::vector<std::shared_ptr<BoundPotential>> local_pots = local_md_pots_->get_potentials();

        intg_->initialize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs, stream);
        for (int i = 1; i <= n_steps; i++) {
            this->_step(local_pots, d_free_idxs, stream);
            if (i % store_x_interval == 0) {
                double *box_ptr = h_box + ((i / store_x_interval) - 1) * 3 * 3;
                double *coord_ptr = h_x + ((i / store_x_interval) - 1) * N_ * 3;
                gpuErrchk(cudaMemcpyAsync(coord_ptr, d_x_t_, N_ * 3 * sizeof(*d_x_t_), cudaMemcpyDeviceToHost, stream));
                gpuErrchk(cudaMemcpyAsync(box_ptr, d_box_t_, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost, stream));
                this->_verify_coords_and_box(coord_ptr, box_ptr, stream);
            }
        }
        intg_->finalize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs, stream);
        local_md_pots_->reset_potentials(stream);
    } catch (...) {
        gpuErrchk(cudaStreamSynchronize(stream));
        gpuErrchk(cudaStreamDestroy(stream));
        throw;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));
}

void Context::multiple_steps_local_selection(
    const int n_steps,
    const int reference_idx,
    const std::vector<int> &selection_idxs,
    const int n_samples,
    const double radius,
    const double k,
    double *h_x,
    double *h_box) {
    const int store_x_interval = n_samples > 0 ? n_steps / n_samples : n_steps + 1;
    if (n_samples < 0) {
        throw std::runtime_error("n_samples < 0");
    }
    if (n_steps % store_x_interval != 0) {
        std::cout << "warning:: n_steps modulo store_x_interval does not equal zero" << std::endl;
    }

    this->_ensure_local_md_intialized();

    cudaStream_t stream;

    // Create stream that doesn't sync with the default stream
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {

        local_md_pots_->setup_from_selection(reference_idx, selection_idxs, radius, k, stream);

        unsigned int *d_free_idxs = local_md_pots_->get_free_idxs();

        std::vector<std::shared_ptr<BoundPotential>> local_pots = local_md_pots_->get_potentials();

        intg_->initialize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs, stream);
        for (int i = 1; i <= n_steps; i++) {
            this->_step(local_pots, d_free_idxs, stream);
            if (i % store_x_interval == 0) {
                double *box_ptr = h_box + ((i / store_x_interval) - 1) * 3 * 3;
                double *coord_ptr = h_x + ((i / store_x_interval) - 1) * N_ * 3;
                gpuErrchk(cudaMemcpyAsync(coord_ptr, d_x_t_, N_ * 3 * sizeof(*d_x_t_), cudaMemcpyDeviceToHost, stream));
                gpuErrchk(cudaMemcpyAsync(box_ptr, d_box_t_, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost, stream));
                this->_verify_coords_and_box(coord_ptr, box_ptr, stream);
            }
        }
        intg_->finalize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs, stream);
        local_md_pots_->reset_potentials(stream);
    } catch (...) {
        gpuErrchk(cudaStreamSynchronize(stream));
        gpuErrchk(cudaStreamDestroy(stream));
        throw;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));
}

void Context::multiple_steps(const int n_steps, const int n_samples, double *h_x, double *h_box) {
    const int store_x_interval = n_samples > 0 ? n_steps / n_samples : n_steps + 1;
    if (n_samples < 0) {
        throw std::runtime_error("n_samples < 0");
    }
    if (n_steps % store_x_interval != 0) {
        std::cout << "warning:: n_steps modulo store_x_interval does not equal zero" << std::endl;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    intg_->initialize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);
    for (int i = 1; i <= n_steps; i++) {
        this->_step(bps_, nullptr, stream);

        if (i % store_x_interval == 0) {
            double *box_ptr = h_box + ((i / store_x_interval) - 1) * 3 * 3;
            double *coord_ptr = h_x + ((i / store_x_interval) - 1) * N_ * 3;
            gpuErrchk(cudaMemcpyAsync(coord_ptr, d_x_t_, N_ * 3 * sizeof(*d_x_t_), cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaMemcpyAsync(box_ptr, d_box_t_, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost, stream));
            this->_verify_coords_and_box(coord_ptr, box_ptr, stream);
        }
    }
    intg_->finalize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream);

    gpuErrchk(cudaStreamSynchronize(stream));
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
    // run any additional movers in this situation.
    // TBD: Handle movers in the local MD case.
    if (d_atom_idxs == nullptr) {
        for (auto mover : movers_) {
            // May modify coords and box size
            mover->move(N_, d_x_t_, d_box_t_, stream);
        }
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

void Context::set_box(const double *in_buffer) {
    gpuErrchk(cudaMemcpy(d_box_t_, in_buffer, 3 * 3 * sizeof(*in_buffer), cudaMemcpyHostToDevice));
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

std::shared_ptr<Integrator> Context::get_integrator() const { return intg_; }

std::vector<std::shared_ptr<BoundPotential>> Context::get_potentials() const { return bps_; }

std::vector<std::shared_ptr<Mover>> Context::get_movers() const { return movers_; }

std::shared_ptr<MonteCarloBarostat<float>> Context::get_barostat() const {
    for (auto mover : movers_) {
        if (is_barostat(mover)) {
            std::shared_ptr<MonteCarloBarostat<float>> baro =
                std::dynamic_pointer_cast<MonteCarloBarostat<float>>(mover);
            return baro;
        }
    }
    return nullptr;
}

} // namespace timemachine
