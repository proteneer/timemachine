// this implements a runner for running bound potentials from python
#pragma once

#include "device_buffer.hpp"
#include "potential.hpp"
#include "stream_manager.hpp"
#include <memory>
#include <vector>

namespace timemachine {

class PotentialExecutor {

public:
    PotentialExecutor(int N, bool parallel = true);

    ~PotentialExecutor();

    void execute_potentials(
        const std::vector<int> param_sizes,
        const std::vector<std::shared_ptr<Potential>> &pots,
        const double *h_x,
        const double *h_box,
        const std::vector<double *> &d_params,
        unsigned long long *h_du_dx,
        unsigned long long *h_du_dp,
        unsigned long long *h_u);

    void du_dp_fixed_to_float(
        const std::vector<int> param_sizes,
        const std::vector<std::shared_ptr<Potential>> &pots,
        const unsigned long long *h_du_dp,
        double *h_du_dp_float);

    int num_atoms() { return N_; };

private:
    const int N_;
    const bool parallel_;

    // Keep buffers that are constant in size regardless of potentials
    DeviceBuffer<double> d_x_;
    DeviceBuffer<double> d_box_;

    DeviceBuffer<unsigned long long> d_du_dx_buffer_;
    DeviceBuffer<unsigned long long> d_u_buffer_;

    StreamManager manager_;
};

} // namespace timemachine
