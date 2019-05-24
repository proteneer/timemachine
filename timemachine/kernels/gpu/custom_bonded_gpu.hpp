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

template <typename RealType>
class HarmonicAngle : public Potential<RealType> {

private:

    int* d_param_idxs_;
    int* d_angle_idxs_;
    int n_angles_;

public:

    HarmonicAngle(
        std::vector<int> param_idxs,
        std::vector<int> angle_idxs
    );

    ~HarmonicAngle();

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


template <typename RealType>
class PeriodicTorsion : public Potential<RealType> {

private:

    int* d_torsion_idxs_;
    int* d_param_idxs_;
    int n_torsions_;

public:

    PeriodicTorsion(
        std::vector<int> torsion_idxs,
        std::vector<int> param_idxs
    );

    ~PeriodicTorsion();

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
