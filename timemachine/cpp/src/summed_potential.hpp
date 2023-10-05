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
    DeviceBuffer<EnergyType> d_u_buffer_;
    StreamManager manager_;

public:
    SummedPotential(
        const std::vector<std::shared_ptr<Potential>> potentials,
        const std::vector<int> params_sizes,
        const bool parallel);

    const std::vector<std::shared_ptr<Potential>> &get_potentials();

    const std::vector<int> &get_parameter_sizes();

    virtual void execute_device(
        const int N,
        const int P,
        const CoordsType *d_x,
        const ParamsType *d_p,
        const CoordsType *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        EnergyType *d_u,
        cudaStream_t stream) override;

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
