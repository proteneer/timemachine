#pragma once

#include "potential.hpp"
#include <memory>
#include <vector>

namespace timemachine {

class FanoutSummedPotential : public Potential {

private:
    const std::vector<std::shared_ptr<Potential>> potentials_;

public:
    FanoutSummedPotential(const std::vector<std::shared_ptr<Potential>> potentials);

    const std::vector<std::shared_ptr<Potential>> &get_potentials();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_du_dl,
        unsigned long long *d_u,
        cudaStream_t stream) override;

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
