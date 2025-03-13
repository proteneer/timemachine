#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "nonbonded_common.hpp"
#include "summed_potential.hpp"
#include <cub/cub.cuh>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace timemachine {

SummedPotential::SummedPotential(
    const std::vector<std::shared_ptr<Potential>> potentials, const std::vector<int> params_sizes, const bool parallel)
    : potentials_(potentials), params_sizes_(params_sizes),
      P_(std::accumulate(params_sizes.begin(), params_sizes.end(), 0)), parallel_(parallel),
      d_u_buffer_(potentials.size()), sum_storage_bytes_(0) {
    if (potentials_.size() != params_sizes_.size()) {
        throw std::runtime_error("number of potentials != number of parameter sizes");
    }

    gpuErrchk(
        cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_.data, d_u_buffer_.data, potentials_.size()));

    gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

SummedPotential::~SummedPotential() { gpuErrchk(cudaFree(d_sum_temp_storage_)); };

const std::vector<std::shared_ptr<Potential>> &SummedPotential::get_potentials() { return potentials_; }

const std::vector<int> &SummedPotential::get_parameter_sizes() { return params_sizes_; }

void SummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    if (P != P_) {
        throw std::runtime_error(
            "SummedPotential::execute_device(): expected " + std::to_string(P_) + " parameters, got " +
            std::to_string(P));
    }
    if (d_u) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_.data, 0, d_u_buffer_.size(), stream));
    }

    int offset = 0;
    if (parallel_) {
        for (auto i = 0; i < potentials_.size(); i++) {
            // Always sync the new streams with the incoming stream to ensure that the state
            // of the incoming buffers are valid
            manager_.sync_from(i, stream);
        }
    }
    cudaStream_t pot_stream = stream;
    for (auto i = 0; i < potentials_.size(); i++) {
        if (parallel_) {
            pot_stream = manager_.get_stream(i);
        }
        potentials_[i]->execute_device(
            N,
            params_sizes_[i],
            d_x,
            d_p + offset,
            d_box,
            d_du_dx,
            d_du_dp == nullptr ? nullptr : d_du_dp + offset,
            d_u == nullptr ? nullptr : d_u_buffer_.data + i,
            pot_stream);

        offset += params_sizes_[i];
        if (parallel_) {
            manager_.sync_to(i, stream);
        }
    }
    if (d_u) {
        gpuErrchk(cub::DeviceReduce::Sum(
            d_sum_temp_storage_, sum_storage_bytes_, d_u_buffer_.data, d_u, potentials_.size(), stream));
    }
};

void SummedPotential::reset() {
    for (auto pot : potentials_) {
        pot.reset();
    }
};

void SummedPotential::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    int offset = 0;

    for (auto i = 0; i < potentials_.size(); i++) {
        potentials_[i]->du_dp_fixed_to_float(N, params_sizes_[i], du_dp + offset, du_dp_float + offset);
        offset += params_sizes_[i];
    }
}

} // namespace timemachine
