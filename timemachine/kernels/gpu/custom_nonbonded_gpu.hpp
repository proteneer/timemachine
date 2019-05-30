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
        const RealType *d_coords,
        const RealType *d_params,
        RealType *d_E,
        RealType *d_dE_dx,
        RealType *d_d2E_dx2,
        // parameter derivatives
        const int num_dp,
        const int *d_param_gather_idxs,
        RealType *d_dE_dp,
        RealType *d_d2E_dxdp) const override;


};


template <typename RealType>
class Electrostatics : public Potential<RealType> {

private:

    RealType* d_scale_matrix_;
    int* d_param_idxs_;

public:

    Electrostatics(
        std::vector<RealType> scale_matrix,
        std::vector<int> param_idxs
    );

    ~Electrostatics();

    virtual void derivatives_device(
        const int num_confs,
        const int num_atoms,
        const RealType *d_coords,
        const RealType *d_params,
        RealType *d_E,
        RealType *d_dE_dx,
        RealType *d_d2E_dx2,
        // parameter derivatives
        const int num_dp,
        const int *d_param_gather_idxs,
        RealType *d_dE_dp,
        RealType *d_d2E_dxdp) const override;


};


}