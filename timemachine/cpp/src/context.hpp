#pragma once


#include <vector>
#include "optimizer.hpp"
#include "potential.hpp"


namespace timemachine {

// a context is a triple of <System, State, Parameters>
// the context does take ownership of any of the input arguments

// this is meant to be one-shot. You are not allowed to set any internals
// once this context has been initialized. This is for safety and ensuring
// that the internal derivatives are kept in sync.

template <typename RealType>
class Context {

private:

    const std::vector<Potential<RealType>*> system_;
    const Optimizer<RealType> *optimizer_;

    RealType *d_params_; // these are really immutable
    int *d_gather_param_idxs_; // these are really immutable

    RealType *d_x_t_;
    RealType *d_v_t_;
    RealType *d_dx_dp_t_;
    RealType *d_dv_dp_t_;

    RealType *d_E_;
    RealType *d_dE_dx_;
    RealType *d_dE_dp_;
    RealType *d_d2E_dx2_;
    RealType *d_d2E_dxdp_;

    int step_;
    int N_;
    int P_;
    int D_;
    int DP_;

public:

    Context(
        const std::vector<Potential<RealType>* > system,
        const Optimizer<RealType> *optimizer,
        const RealType *h_params,
        const RealType *h_x0,
        const RealType *h_v0,
        const int N,
        const int D,
        const int P,
        const int *h_param_gather_idxs,
        const int DP);

    int num_atoms() const { return N_; };

    int num_dims() const { return D_; };

    int num_params() const { return P_; };

    int num_dparams() const { return DP_; };

    void step();

    void get_E(RealType *buffer) const;

    void get_dE_dx(RealType *buffer) const;

    void get_dE_dp(RealType *buffer) const;

    void get_x(RealType *buffer) const;

    void get_v(RealType *buffer) const;

    void get_dx_dp(RealType *buffer) const;

    void get_dv_dp(RealType *buffer) const;

    ~Context();

};

}