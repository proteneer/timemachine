#include "context.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include <chrono>
#include <cub/cub.cuh>
#include <iostream>

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
    gpuErrchk(cudaMalloc(&d_du_dx_t_, N * 3 * sizeof(*d_du_dx_t_)));
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
    gpuErrchk(cudaFree(d_du_dx_t_));
    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_storage_));

    // for(int i=0; i < streams_.size(); i++) {
    // gpuErrchk(cudaStreamDestroy(streams_[i]));
    // }
};

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

        for (int i = 1; i <= lambda_schedule.size(); i++) {
            // decide if we need to store the du_dl for this step
            unsigned long long *du_dl_ptr = nullptr;
            if (i % store_du_dl_interval == 0) {
                // pemdas but just to make it clear we're doing pointer arithmetic
                du_dl_ptr = d_du_dl_buffer + ((i / store_du_dl_interval) - 1);
            }

            double lambda = lambda_schedule[i - 1];
            this->_step(bps_, lambda, du_dl_ptr, stream);

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
        for (int step = 1; step <= n_steps; step++) {

            this->_step(bps_, lambda, nullptr, stream);

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
    this->_step(bps_, lambda, nullptr, stream);
    gpuErrchk(cudaDeviceSynchronize());
}

void Context::_step(
    std::vector<BoundPotential *> &bps, const double lambda, unsigned long long *du_dl_out, const cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(d_du_dx_t_, 0, N_ * 3 * sizeof(*d_du_dx_t_), stream));

    if (du_dl_out) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, N_ * sizeof(*d_du_dl_buffer_), stream));
    }

    for (int i = 0; i < bps.size(); i++) {
        bps[i]->execute_device(
            N_,
            d_x_t_,
            d_box_t_,
            lambda,
            d_du_dx_t_, // we only need the forces
            nullptr,
            du_dl_out ? d_du_dl_buffer_ : nullptr,
            nullptr,
            stream);
    }

    // compute du_dl
    if (du_dl_out) {
        cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_du_dl_buffer_, du_dl_out, N_, stream);
        gpuErrchk(cudaPeekAtLastError());
    }

    // for(int i=0; i < streams_.size(); i++) {
    // gpuErrchk(cudaStreamSynchronize(streams_[i]));
    // }
    intg_->step_fwd(d_x_t_, d_v_t_, d_du_dx_t_, d_box_t_, stream);

    if (barostat_) {
        // May modify coords, du_dx and box size
        barostat_->inplace_move(d_x_t_, d_box_t_, lambda, stream);
    }

    step_ += 1;
};

int Context::num_atoms() const { return N_; }

void Context::set_x_t(const double *in_buffer) {
    gpuErrchk(cudaMemcpy(d_x_t_, in_buffer, N_ * 3 * sizeof(*in_buffer), cudaMemcpyHostToDevice));
}

void Context::get_du_dx_t_minus_1(unsigned long long *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_du_dx_t_, N_ * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
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
