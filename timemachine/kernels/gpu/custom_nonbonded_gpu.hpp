#pragma once

#include "potential.hpp"
#include <vector>


namespace timemachine {

template <typename RealType>
class LennardJones : public Potential<RealType> {

private:

    RealType* d_scale_matrix_;
    int* d_param_idxs_;

public:

    LennardJones(
        std::vector<RealType> scale_matrix,
        std::vector<int> param_idxs
    );

    ~LennardJones();

    virtual void derivatives_device(
        const int num_confs,
        const int num_atoms,
        const int num_params,
        const RealType *d_coords,
        const RealType *d_params,
        RealType *d_E,
        RealType *d_dE_dx,
        // parameter derivatives
        const RealType *d_dx_dp,
        const int *d_dp_idxs,
        const int num_dp_idxs,
        RealType *d_dE_dp,
        RealType *d_d2E_dxdp) const override;

};


}