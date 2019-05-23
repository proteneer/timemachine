#pragma once

#include<vector>

namespace timemachine {

template <typename RealType>
class Potential {

public:

    /*

    Convenience function that wraps around the derivatives_device by allocating,
    transferring, and deleting gpu and cpu buffers as needed.

    */
    void derivatives_host(
        const int num_atoms,
        const int num_params,
        const RealType *h_coords,
        const RealType *h_params,
        RealType *h_E,
        RealType *h_dE_dx,

        const RealType *h_dx_dp,
        const int *h_dp_idxs,
        const int num_dp_idxs,
        RealType *h_dE_dp,
        RealType *h_d2E_dxdp) const;
    /*

    Computes the various derivatives of the energy with respect to the arguments.

    */
    virtual void derivatives_device(
        const int num_atoms,
        const int num_params,
        const RealType *d_coords,
        const RealType *d_params,
        RealType *d_E,
        RealType *d_dE_dx,

        const RealType *d_dx_dp,
        const int *d_dp_idxs,
        const int num_dp_idxs,
        RealType *d_dE_dp,
        RealType *d_d2E_dxdp) const = 0;

};

}