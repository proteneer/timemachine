#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType>
class HarmonicBond : public Potential<RealType> {

private:

    int* d_bond_idxs_;
    int* d_param_idxs_;
    int n_bonds_;

public:

    HarmonicBond(
        std::vector<int> bond_idxs,
        std::vector<int> param_idxs
    );

    ~HarmonicBond();

    virtual void derivatives_device(
        const int num_atoms,
        const int num_params,
        const RealType *d_coords,
        const RealType *d_params,
        const RealType *d_dxdps,
        RealType *d_E,
        RealType *d_dE_dp,
        RealType *d_dE_dx,
        RealType *d_d2E_dxdp) const override;

};


}