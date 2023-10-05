#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class CentroidRestraint : public Potential {

private:
    int *d_group_a_idxs_;
    int *d_group_b_idxs_;

    unsigned long long *d_centroid_a_;
    unsigned long long *d_centroid_b_;

    EnergyType *d_u_buffer_;

    int N_A_;
    int N_B_;

    double kb_;
    double b0_;

public:
    CentroidRestraint(
        const std::vector<int> &group_a_idxs, const std::vector<int> &group_b_idxs, const double kb, const double b0);

    ~CentroidRestraint();

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
};

} // namespace timemachine
