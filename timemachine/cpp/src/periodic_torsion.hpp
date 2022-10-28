#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class PeriodicTorsion : public Potential {

private:
    int *d_torsion_idxs_;

    const int T_;

public:
    int num_torsions() const { return T_; }

    PeriodicTorsion(const std::vector<int> &torsion_idxs);

    ~PeriodicTorsion();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
