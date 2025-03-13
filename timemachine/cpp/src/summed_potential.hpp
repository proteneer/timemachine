#pragma once

#include "device_buffer.hpp"
#include "potential.hpp"
#include "stream_manager.hpp"
#include <memory>
#include <vector>

namespace timemachine {

class SummedPotential : public Potential {

private:
    const std::vector<std::shared_ptr<Potential>> potentials_;
    const std::vector<int> params_sizes_;
    const int P_; // sum(params_sizes)
    const bool parallel_;
    DeviceBuffer<__int128> d_u_buffer_;
    StreamManager manager_;

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

public:
    SummedPotential(
        const std::vector<std::shared_ptr<Potential>> potentials,
        const std::vector<int> params_sizes,
        const bool parallel);

    ~SummedPotential();

    const std::vector<std::shared_ptr<Potential>> &get_potentials();

    const std::vector<int> &get_parameter_sizes();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        __int128 *d_u,
        cudaStream_t stream) override;

    virtual void reset() override;

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
