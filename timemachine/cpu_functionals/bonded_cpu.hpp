#pragma once
#include <array>
#include <iostream>
#include <complex>
#include <vector>
#include "utils.hpp"

namespace timemachine {

template <typename NumericType>
class HarmonicBond {

public:

    const std::vector<NumericType> params_;
    const std::vector<size_t> param_idxs_;
    const std::vector<size_t> bond_idxs_;

public:

    HarmonicBond(
        std::vector<NumericType> params,
        std::vector<size_t> param_idxs,
        std::vector<size_t> bond_idxs
    ) : params_(params), 
        param_idxs_(param_idxs),
        bond_idxs_(bond_idxs) {};


    NumericType energy(
        const std::array<NumericType, 6> &xs,
        const std::array<NumericType, 2> &params) const {
        NumericType x0 = xs[0];
        NumericType y0 = xs[1];
        NumericType z0 = xs[2];
        NumericType x1 = xs[3];
        NumericType y1 = xs[4];
        NumericType z1 = xs[5];
        NumericType kb = params[0];
        NumericType b0 = params[1];

        auto dx = x0-x1;
        auto dy = y0-y1;
        auto dz = z0-z1;
        auto db = sqrt(dx*dx+dy*dy+dz*dz)-b0;
        return kb/2.0*db*db;
    }

    // gradient of a single interaction.
    template<
        typename CoordType,
        typename ParamType,
        typename OutType>
    void ixn_grad(
        const std::array<CoordType, 6> &xs,
        const std::array<ParamType, 2> &params,
        std::array<OutType, 6> &dxs) const {
        CoordType x0 = xs[0];
        CoordType y0 = xs[1];
        CoordType z0 = xs[2];
        CoordType x1 = xs[3];
        CoordType y1 = xs[4];
        CoordType z1 = xs[5];
        ParamType kb = params[0];
        ParamType b0 = params[1];

        CoordType dx = x0-x1;
        CoordType dy = y0-y1;
        CoordType dz = z0-z1;
        CoordType dij = sqrt(dx*dx+dy*dy+dz*dz);
        auto db = dij-b0;
        dxs[0] = kb*db*dx/dij;
        dxs[1] = kb*db*dy/dij;
        dxs[2] = kb*db*dz/dij;
        dxs[3] = -dxs[0];
        dxs[4] = -dxs[1];
        dxs[5] = -dxs[2];
    }

    size_t numBonds() const {
        return bond_idxs_.size()/2;
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

        for(size_t i=0; i < numBonds(); i++) {

            size_t s_atom_idx = bond_idxs_[i*2+0];
            size_t e_atom_idx = bond_idxs_[i*2+1];
            size_t p_k_idx = param_idxs_[i*2+0];
            size_t p_b_idx = param_idxs_[i*2+1];
            NumericType kb = params_[p_k_idx];
            NumericType b0 = params_[p_b_idx];

            if(s_atom_idx == e_atom_idx) {
                throw std::runtime_error("self-bond detected.");
            }

            const NumericType x0 = coords[s_atom_idx*3+0];
            const NumericType y0 = coords[s_atom_idx*3+1];
            const NumericType z0 = coords[s_atom_idx*3+2];
            const NumericType x1 = coords[e_atom_idx*3+0];
            const NumericType y1 = coords[e_atom_idx*3+1];
            const NumericType z1 = coords[e_atom_idx*3+2];

            std::array<NumericType, 6> xs({x0, y0, z0, x1, y1, z1});
            std::array<NumericType, 2> params({kb, b0});

            *energy_out += energy(xs, params);

            std::array<NumericType, 6> dxs;
            ixn_grad<NumericType, NumericType>(xs, params, dxs);
            grad_out[s_atom_idx*3 + 0] += dxs[0];
            grad_out[s_atom_idx*3 + 1] += dxs[1];
            grad_out[s_atom_idx*3 + 2] += dxs[2];
            grad_out[e_atom_idx*3 + 0] += dxs[3];
            grad_out[e_atom_idx*3 + 1] += dxs[4];
            grad_out[e_atom_idx*3 + 2] += dxs[5];

            NumericType step = 1e-100;

            // compute mixed partials, loop over all the parameters
            // (ytz) TODO: unroll this
            for(size_t j=0; j < 2; j++) {
                std::array<std::complex<NumericType>, 2> cparams = timemachine::convert_to_complex(params);
                cparams[j] = std::complex<NumericType>(cparams[j].real(), step);

                std::array<std::complex<NumericType>, 6> dcxs;
                ixn_grad<NumericType, std::complex<NumericType> >(xs, cparams, dcxs);

                std::array<NumericType, 6> ddxs;
                for(int k=0; k < 6; k++) {
                    ddxs[k] = dcxs[k].imag() / step;
                }
                size_t p_idx = param_idxs_[i*2+j];
                total_out[p_idx*n_atoms*3 + s_atom_idx*3 + 0] += ddxs[0];
                total_out[p_idx*n_atoms*3 + s_atom_idx*3 + 1] += ddxs[1];
                total_out[p_idx*n_atoms*3 + s_atom_idx*3 + 2] += ddxs[2];
                total_out[p_idx*n_atoms*3 + e_atom_idx*3 + 0] += ddxs[3];
                total_out[p_idx*n_atoms*3 + e_atom_idx*3 + 1] += ddxs[4];
                total_out[p_idx*n_atoms*3 + e_atom_idx*3 + 2] += ddxs[5];
            }

            // compute hessian vector product, looping over all atoms and parameters
            std::array<size_t, 6> indices({
                s_atom_idx*3+0,
                s_atom_idx*3+1,
                s_atom_idx*3+2,
                e_atom_idx*3+0,
                e_atom_idx*3+1,
                e_atom_idx*3+2,
            });

            for(size_t j=0; j < 6; j++) {
                std::array<std::complex<NumericType>, 6> cxs = timemachine::convert_to_complex(xs);
                cxs[j] = std::complex<NumericType>(cxs[j].real(), step);

                std::array<std::complex<NumericType>, 6> dcxs;
                ixn_grad<std::complex<NumericType>, NumericType >(cxs, params, dcxs);

                std::array<NumericType, 6> ddxs;
                for(int k=0; k < 6; k++) {
                    ddxs[k] = dcxs[k].imag() / step;
                }
                for(size_t p=0; p < 2; p++) {
                    size_t p_idx = param_idxs_[i*2+p];
                    auto dx0 = ddxs[0] * dxdp[p_idx*n_atoms*3 + s_atom_idx*3 + 0] + ddxs[3] * dxdp[p_idx*n_atoms*3 + e_atom_idx*3 + 0];
                    auto dx1 = ddxs[1] * dxdp[p_idx*n_atoms*3 + s_atom_idx*3 + 1] + ddxs[4] * dxdp[p_idx*n_atoms*3 + e_atom_idx*3 + 1];
                    auto dx2 = ddxs[2] * dxdp[p_idx*n_atoms*3 + s_atom_idx*3 + 2] + ddxs[5] * dxdp[p_idx*n_atoms*3 + e_atom_idx*3 + 2];
                    total_out[p_idx*n_atoms*3 + indices[j]] += dx0+dx1+dx2;
                }
            }
        }
    }
};

}