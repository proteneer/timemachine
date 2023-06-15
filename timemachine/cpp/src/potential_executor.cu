#include "gpu_utils.cuh"
#include "potential_executor.hpp"

const static int D = 3;

namespace timemachine {

PotentialExecutor::PotentialExecutor(int N, bool parallel)
    : N_(N), parallel_(parallel), d_x_(N * D), d_box_(D * D), d_du_dx_buffer_(N * D), d_u_buffer_(N){};

PotentialExecutor::~PotentialExecutor() {}

void PotentialExecutor::execute_bound(
    const std::vector<std::shared_ptr<BoundPotential>> &bps,
    const double *h_x,
    const double *h_box,
    unsigned long long *h_du_dx,
    unsigned long long *h_u) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    d_x_.copy_from(h_x);
    d_box_.copy_from(h_box);

    if (h_du_dx != nullptr) {
        gpuErrchk(cudaMemsetAsync(d_du_dx_buffer_.data, 0, d_du_dx_buffer_.size, stream));
    }

    if (h_u != nullptr) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_.data, 0, d_u_buffer_.size, stream));
    }

    if (parallel_) {
        for (int i = 0; i < bps.size(); i++) {
            // Always sync the new streams with the incoming stream to ensure that the state
            // of the incoming buffers are valid
            manager_.sync_from(i, stream);
        }
    }
    cudaStream_t pot_stream = stream;
    for (int i = 0; i < bps.size(); i++) {
        if (parallel_) {
            pot_stream = manager_.get_stream(i);
        }
        bps[i]->execute_device(
            N_,
            d_x_.data,
            d_box_.data,
            h_du_dx == nullptr ? nullptr : d_du_dx_buffer_.data,
            nullptr,
            h_u == nullptr ? nullptr : d_u_buffer_.data,
            pot_stream);
        if (parallel_) {
            manager_.sync_to(i, stream);
        }
    }

    if (h_du_dx != nullptr) {
        gpuErrchk(cudaMemcpyAsync(h_du_dx, d_du_dx_buffer_.data, d_du_dx_buffer_.size, cudaMemcpyDeviceToHost, stream));
    }

    if (h_u != nullptr) {
        gpuErrchk(cudaMemcpyAsync(h_u, d_u_buffer_.data, d_u_buffer_.size, cudaMemcpyDeviceToHost, stream));
    }

    gpuErrchk(cudaStreamSynchronize(stream));
};

void PotentialExecutor::execute_potentials(
    const std::vector<std::shared_ptr<Potential>> &pots,
    const double *h_x,
    const double *h_box,
    const std::vector<int> &param_sizes,
    const DeviceBuffer<double> &d_params,
    unsigned long long *h_du_dx,
    unsigned long long *h_du_dp,
    unsigned long long *h_u) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    d_x_.copy_from(h_x);
    d_box_.copy_from(h_box);

    if (h_du_dx != nullptr) {
        gpuErrchk(cudaMemsetAsync(d_du_dx_buffer_.data, 0, d_du_dx_buffer_.size, stream));
    }

    if (h_u != nullptr) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_.data, 0, d_u_buffer_.size, stream));
    }

    std::unique_ptr<DeviceBuffer<unsigned long long>> du_dp_buffer;
    if (h_du_dp != nullptr) {
        du_dp_buffer.reset(new DeviceBuffer<unsigned long long>(d_params.length));
        gpuErrchk(cudaMemsetAsync(du_dp_buffer->data, 0, du_dp_buffer->size, stream));
    }

    int offset = 0;
    if (parallel_) {
        for (int i = 0; i < pots.size(); i++) {
            // Always sync the new streams with the incoming stream to ensure that the state
            // of the incoming buffers are valid
            manager_.sync_from(i, stream);
        }
    }
    cudaStream_t pot_stream = stream;
    for (int i = 0; i < pots.size(); i++) {
        if (parallel_) {
            pot_stream = manager_.get_stream(i);
        }
        int param_size = param_sizes[i];
        pots[i]->execute_device(
            N_,
            param_size,
            d_x_.data,
            param_size > 0 ? d_params.data + offset : nullptr,
            d_box_.data,
            h_du_dx == nullptr ? nullptr : d_du_dx_buffer_.data,
            h_du_dp == nullptr ? nullptr : du_dp_buffer->data + offset,
            h_u == nullptr ? nullptr : d_u_buffer_.data,
            pot_stream);

        offset += param_size;
        if (parallel_) {
            manager_.sync_to(i, stream);
        }
    }

    if (h_du_dx != nullptr) {
        gpuErrchk(cudaMemcpyAsync(h_du_dx, d_du_dx_buffer_.data, d_du_dx_buffer_.size, cudaMemcpyDeviceToHost, stream));
    }

    if (h_du_dp != nullptr) {
        gpuErrchk(cudaMemcpyAsync(h_du_dp, du_dp_buffer->data, du_dp_buffer->size, cudaMemcpyDeviceToHost, stream));
    }

    if (h_u != nullptr) {
        gpuErrchk(cudaMemcpyAsync(h_u, d_u_buffer_.data, d_u_buffer_.size, cudaMemcpyDeviceToHost, stream));
    }

    gpuErrchk(cudaStreamSynchronize(stream));
};

void PotentialExecutor::du_dp_fixed_to_float(
    const std::vector<int> &param_sizes,
    const std::vector<std::shared_ptr<Potential>> &pots,
    const unsigned long long *h_du_dp,
    double *h_du_dp_float) {
    int offset = 0;

    for (int i = 0; i < pots.size(); i++) {
        int bp_size = param_sizes[i];
        pots[i]->du_dp_fixed_to_float(N_, bp_size, h_du_dp + offset, h_du_dp_float + offset);
        offset += bp_size;
    }
}

} // namespace timemachine
