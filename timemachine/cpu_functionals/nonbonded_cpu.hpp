#pragma once
#include <array>
#include <iostream>
#include <complex>
#include <vector>
#include "utils.hpp"
#include "constants.hpp"
namespace timemachine {

template <typename NumericType>
class Electrostatics {

private:

    const std::vector<NumericType> params_;
    const std::vector<size_t> global_param_idxs_;
    const std::vector<size_t> param_idxs_;
    const std::vector<NumericType> scale_matrix_;


public:

    Electrostatics(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<NumericType> scale_matrix
    ) : params_(params),
        global_param_idxs_(global_param_idxs),
        param_idxs_(param_idxs),
        scale_matrix_(scale_matrix) {};


    NumericType ixn_energy(
        const std::array<NumericType, 6> &xs,
        const std::array<NumericType, 2> &params) const {
        NumericType x0 = xs[0];
        NumericType y0 = xs[1];
        NumericType z0 = xs[2];

        NumericType x1 = xs[3];
        NumericType y1 = xs[4];
        NumericType z1 = xs[5];

        NumericType q0 = params[0];
        NumericType q1 = params[1];

        NumericType dx = x0 - x1;
        NumericType dy = y0 - y1;
        NumericType dz = z0 - z1;

        NumericType dij = timemachine::norm(dx, dy, dz);

        return (ONE_4PI_EPS0*q0*q1)/dij;
    }

    template<
        typename CoordType,
        typename ParamType,
        typename OutType>
    std::array<OutType, 6> ixn_gradient(
        const std::array<CoordType, 6> &xs,
        const std::array<ParamType, 2> &params) const {
        CoordType x0 = xs[0];
        CoordType y0 = xs[1];
        CoordType z0 = xs[2];

        CoordType x1 = xs[3];
        CoordType y1 = xs[4];
        CoordType z1 = xs[5];

        ParamType q0 = params[0];
        ParamType q1 = params[1];

        CoordType dx = x0 - x1;
        CoordType dy = y0 - y1;
        CoordType dz = z0 - z1;

        CoordType dij = timemachine::norm(dx, dy, dz);
        CoordType d3ij = dij*dij*dij;

        OutType PREFACTOR = ONE_4PI_EPS0*q0*q1/d3ij;

        std::array<OutType, 6> grads;

        grads[0*3 + 0] = PREFACTOR*(-dx);
        grads[0*3 + 1] = PREFACTOR*(-dy);
        grads[0*3 + 2] = PREFACTOR*(-dz);

        grads[1*3 + 0] = PREFACTOR*(dx);
        grads[1*3 + 1] = PREFACTOR*(dy);
        grads[1*3 + 2] = PREFACTOR*(dz);

        return grads;
    }

    /*
    Implements the total derivative of the gradient at a point in time. We
    use raw pointers to make it easier to interface with the wrapper layers and
    to maintain some level of consistency with the C++ code.
    */
    void total_derivative(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* coords, // [N, 3]
        const NumericType* dxdp, // [P, N, 3]
        NumericType* energy_out, // []
        NumericType* grad_out, // [N,3]
        NumericType* total_out // [P, N, 3]
    ) const {

        // use upper right symmetric later
        for(size_t atom_0_idx=0; atom_0_idx < n_atoms; atom_0_idx++) {

            const NumericType x0 = coords[atom_0_idx*3+0];
            const NumericType y0 = coords[atom_0_idx*3+1];
            const NumericType z0 = coords[atom_0_idx*3+2];
            const NumericType q0 = params_[param_idxs_[atom_0_idx]];

            for(size_t atom_1_idx=atom_0_idx+1; atom_1_idx < n_atoms; atom_1_idx++ ) {

                NumericType scale = scale_matrix_[atom_0_idx*n_atoms + atom_1_idx];
                if(scale == 0) {
                    continue;
                }

                const NumericType x1 = coords[atom_1_idx*3+0];
                const NumericType y1 = coords[atom_1_idx*3+1];
                const NumericType z1 = coords[atom_1_idx*3+2];
                const NumericType q1 = params_[param_idxs_[atom_1_idx]];

                std::array<NumericType, 6> xs({x0, y0, z0, x1, y1, z1});
                std::array<NumericType, 2> params({q0, q1});

                *energy_out += scale*ixn_energy(xs, params);

                std::array<NumericType, 6> grads = ixn_gradient<NumericType, NumericType, NumericType>(xs, params);

                grad_out[atom_0_idx*3 + 0] += scale*grads[0];
                grad_out[atom_0_idx*3 + 1] += scale*grads[1];
                grad_out[atom_0_idx*3 + 2] += scale*grads[2];

                grad_out[atom_1_idx*3 + 0] += scale*grads[3];
                grad_out[atom_1_idx*3 + 1] += scale*grads[4];
                grad_out[atom_1_idx*3 + 2] += scale*grads[5];

                NumericType step = 1e-50;

                // compute mixed partials, loop over all the parameters
                // (ytz) TODO: unroll this
                std::array<size_t, 2> pidx({param_idxs_[atom_0_idx], param_idxs_[atom_1_idx]});
                for(size_t j=0; j < 2; j++) {
                    std::array<std::complex<NumericType>, 2> cparams = timemachine::convert_to_complex(params);
                    cparams[j] = std::complex<NumericType>(cparams[j].real(), step);
                    std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<NumericType, std::complex<NumericType>, std::complex<NumericType> >(xs, cparams);

                    std::array<NumericType, 6> ddxs;
                    for(int k=0; k < 6; k++) {
                        ddxs[k] = dcxs[k].imag() / step;
                    }

                    size_t p_idx = global_param_idxs_[pidx[j]];

                    total_out[p_idx*n_atoms*3 + atom_0_idx*3 + 0] += scale*ddxs[0];
                    total_out[p_idx*n_atoms*3 + atom_0_idx*3 + 1] += scale*ddxs[1];
                    total_out[p_idx*n_atoms*3 + atom_0_idx*3 + 2] += scale*ddxs[2];
                    total_out[p_idx*n_atoms*3 + atom_1_idx*3 + 0] += scale*ddxs[3];
                    total_out[p_idx*n_atoms*3 + atom_1_idx*3 + 1] += scale*ddxs[4];
                    total_out[p_idx*n_atoms*3 + atom_1_idx*3 + 2] += scale*ddxs[5];
                }

                std::array<size_t, 6> indices({
                    atom_0_idx*3+0,
                    atom_0_idx*3+1,
                    atom_0_idx*3+2,
                    atom_1_idx*3+0,
                    atom_1_idx*3+1,
                    atom_1_idx*3+2
                });

                for(size_t j=0; j < 6; j++) {

                    std::array<std::complex<NumericType>, 6> cxs = timemachine::convert_to_complex(xs);
                    cxs[j] = std::complex<NumericType>(cxs[j].real(), step);
                    std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<std::complex<NumericType>, NumericType, std::complex<NumericType> >(cxs, params);
                    std::array<NumericType, 6> ddxs;
                    for(int k=0; k < 6; k++) {
                        ddxs[k] = scale*dcxs[k].imag() / step;
                    }

                    for(size_t p=0; p < n_params; p++) {
                        auto dx0 = ddxs[0] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 0] + ddxs[3] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 0];
                        auto dx1 = ddxs[1] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 1] + ddxs[4] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 1];
                        auto dx2 = ddxs[2] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 2] + ddxs[5] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 2];
                        total_out[p*n_atoms*3 + indices[j]] += dx0+dx1+dx2;
                    }
                }

            }
        }
    }
};

template <typename NumericType>
class LennardJones {

private:

    const std::vector<NumericType> params_;
    const std::vector<size_t> global_param_idxs_;
    const std::vector<size_t> param_idxs_; // [N, 2] for sig eps
    const std::vector<NumericType> scale_matrix_;


public:

    LennardJones(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<NumericType> scale_matrix
    ) : params_(params), 
        global_param_idxs_(global_param_idxs),
        param_idxs_(param_idxs),
        scale_matrix_(scale_matrix) {};

    // unscaled interaction energies
    NumericType ixn_energy(
        const std::array<NumericType, 6> &xs,
        const std::array<NumericType, 4> &params) const {
        NumericType x0 = xs[0];
        NumericType y0 = xs[1];
        NumericType z0 = xs[2];

        NumericType x1 = xs[3];
        NumericType y1 = xs[4];
        NumericType z1 = xs[5];

        NumericType sig0 = params[0];
        NumericType eps0 = params[1];
        NumericType sig1 = params[2];
        NumericType eps1 = params[3];

        NumericType dx = x0 - x1;
        NumericType dy = y0 - y1;
        NumericType dz = z0 - z1;

        NumericType dij = timemachine::norm(dx, dy, dz);

        NumericType eps = sqrt(eps0 * eps1);
        NumericType sig = (sig0 + sig1)/2;

        NumericType sig2 = sig/dij;
        sig2 *= sig2;
        NumericType sig6 = sig2*sig2*sig2;

        NumericType energy = 4*eps*(sig6-1.0)*sig6;

        return energy;
    }

    template<
        typename CoordType,
        typename ParamType,
        typename OutType>
    std::array<OutType, 6> ixn_gradient(
        const std::array<CoordType, 6> &xs,
        const std::array<ParamType, 4> &params) const {
        CoordType x0 = xs[0];
        CoordType y0 = xs[1];
        CoordType z0 = xs[2];

        CoordType x1 = xs[3];
        CoordType y1 = xs[4];
        CoordType z1 = xs[5];

        CoordType dx = x0 - x1;
        CoordType dy = y0 - y1;
        CoordType dz = z0 - z1;

        ParamType sig0 = params[0];
        ParamType eps0 = params[1];
        ParamType sig1 = params[2];
        ParamType eps1 = params[3];

        ParamType eps = sqrt(eps0 * eps1);
        ParamType sig = (sig0 + sig1)/2;

        ParamType sig2 = sig*sig;
        ParamType sig3 = sig2*sig;
        ParamType sig5 = sig3*sig2;
        ParamType sig6 = sig5*sig;
        ParamType sig12 = sig6*sig6;

        // rij = d2ij, avoids a square root.
        CoordType rij = dx*dx + dy*dy + dz*dz;
        CoordType rij3 = rij * rij * rij;
        CoordType rij4 = rij3 * rij;
        CoordType rij7 = rij4 * rij3;

        // (ytz): 99 % sure this loses precision so we need to refactor
        OutType sig12rij7 = sig12/rij7;
        OutType sig6rij4 = sig6/rij4;

        OutType dEdx = 4*eps*(sig12rij7*12*dx - sig6rij4*6*dx);
        OutType dEdy = 4*eps*(sig12rij7*12*dy - sig6rij4*6*dy);
        OutType dEdz = 4*eps*(sig12rij7*12*dz - sig6rij4*6*dz);

        std::array<OutType, 6> grads;

        grads[0*3+0] = -dEdx;
        grads[0*3+1] = -dEdy;
        grads[0*3+2] = -dEdz;

        grads[1*3+0] = dEdx;
        grads[1*3+1] = dEdy;
        grads[1*3+2] = dEdz;

        return grads;
    }

    /*
    Implements the total derivative of the gradient at a point in time. We
    use raw pointers to make it easier to interface with the wrapper layers and
    to maintain some level of consistency with the C++ code.
    */
    void total_derivative(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* coords, // [N, 3]
        const NumericType* dxdp, // [P, N, 3]
        NumericType* energy_out, // []
        NumericType* grad_out, // [N,3]
        NumericType* total_out // [P, N, 3]
    ) const {

        // use upper right symmetric later
        for(size_t atom_0_idx=0; atom_0_idx < n_atoms; atom_0_idx++) {

            const NumericType x0 = coords[atom_0_idx*3+0];
            const NumericType y0 = coords[atom_0_idx*3+1];
            const NumericType z0 = coords[atom_0_idx*3+2];
            const NumericType sig0 = params_[param_idxs_[atom_0_idx*2+0]];
            const NumericType eps0 = params_[param_idxs_[atom_0_idx*2+1]];

            for(size_t atom_1_idx=atom_0_idx+1; atom_1_idx < n_atoms; atom_1_idx++ ) {

                NumericType scale = scale_matrix_[atom_0_idx*n_atoms + atom_1_idx];
                if(scale == 0) {
                    continue;
                }

                const NumericType x1 = coords[atom_1_idx*3+0];
                const NumericType y1 = coords[atom_1_idx*3+1];
                const NumericType z1 = coords[atom_1_idx*3+2];
                const NumericType sig1 = params_[param_idxs_[atom_1_idx*2+0]];
                const NumericType eps1 = params_[param_idxs_[atom_1_idx*2+1]];

                std::array<NumericType, 6> xs({x0, y0, z0, x1, y1, z1});
                std::array<NumericType, 4> params({sig0, eps0, sig1, eps1});

                *energy_out += scale*ixn_energy(xs, params);

                std::array<NumericType, 6> grads = ixn_gradient<NumericType, NumericType, NumericType>(xs, params);

                grad_out[atom_0_idx*3 + 0] += scale*grads[0];
                grad_out[atom_0_idx*3 + 1] += scale*grads[1];
                grad_out[atom_0_idx*3 + 2] += scale*grads[2];

                grad_out[atom_1_idx*3 + 0] += scale*grads[3];
                grad_out[atom_1_idx*3 + 1] += scale*grads[4];
                grad_out[atom_1_idx*3 + 2] += scale*grads[5];

                NumericType step = 1e-50;

                // compute mixed partials, loop over all the parameters
                // (ytz) TODO: unroll this
                std::array<size_t, 4> pidx({
                    param_idxs_[atom_0_idx*2+0],
                    param_idxs_[atom_0_idx*2+1],
                    param_idxs_[atom_1_idx*2+0],
                    param_idxs_[atom_1_idx*2+1]
                });

                for(size_t j=0; j < 4; j++) {
                    std::array<std::complex<NumericType>, 4> cparams = timemachine::convert_to_complex(params);
                    cparams[j] = std::complex<NumericType>(cparams[j].real(), step);
                    std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<NumericType, std::complex<NumericType>, std::complex<NumericType> >(xs, cparams);

                    std::array<NumericType, 6> ddxs;
                    for(int k=0; k < 6; k++) {
                        ddxs[k] = dcxs[k].imag() / step;
                    }

                    size_t p_idx = global_param_idxs_[pidx[j]];

                    total_out[p_idx*n_atoms*3 + atom_0_idx*3 + 0] += scale*ddxs[0];
                    total_out[p_idx*n_atoms*3 + atom_0_idx*3 + 1] += scale*ddxs[1];
                    total_out[p_idx*n_atoms*3 + atom_0_idx*3 + 2] += scale*ddxs[2];
                    total_out[p_idx*n_atoms*3 + atom_1_idx*3 + 0] += scale*ddxs[3];
                    total_out[p_idx*n_atoms*3 + atom_1_idx*3 + 1] += scale*ddxs[4];
                    total_out[p_idx*n_atoms*3 + atom_1_idx*3 + 2] += scale*ddxs[5];
                }

                std::array<size_t, 6> indices({
                    atom_0_idx*3+0,
                    atom_0_idx*3+1,
                    atom_0_idx*3+2,
                    atom_1_idx*3+0,
                    atom_1_idx*3+1,
                    atom_1_idx*3+2
                });

                for(size_t j=0; j < 6; j++) {

                    std::array<std::complex<NumericType>, 6> cxs = timemachine::convert_to_complex(xs);
                    cxs[j] = std::complex<NumericType>(cxs[j].real(), step);
                    std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<std::complex<NumericType>, NumericType, std::complex<NumericType> >(cxs, params);
                    std::array<NumericType, 6> ddxs;
                    for(int k=0; k < 6; k++) {
                        ddxs[k] = scale*dcxs[k].imag() / step;
                    }

                    for(size_t p=0; p < n_params; p++) {
                        auto dx0 = ddxs[0] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 0] + ddxs[3] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 0];
                        auto dx1 = ddxs[1] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 1] + ddxs[4] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 1];
                        auto dx2 = ddxs[2] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 2] + ddxs[5] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 2];
                        total_out[p*n_atoms*3 + indices[j]] += dx0+dx1+dx2;
                    }
                }

            }
        }
    }
};

} // namespace timemachine