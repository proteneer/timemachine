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
        const RealType *h_dxdps,
        RealType *h_E,
        RealType *h_dE_dp,
        RealType *h_dE_dx,
        RealType *h_d2E_dxdp) const;
    /*

    Computes the various derivatives of the energy with respect to the arguments.

    */
    virtual void derivatives_device(
        const int num_atoms,
        const int num_params,
        const RealType *d_coords,
        const RealType *d_params,
        const RealType *d_dxdps,
        RealType *d_E,
        RealType *d_dE_dp,
        RealType *d_dE_dx,
        RealType *d_d2E_dxdp) const = 0;

};

}