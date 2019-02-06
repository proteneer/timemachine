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
        NumericType* energy_out, // []
        NumericType* grad_out, // [N,3]
        NumericType* hessian_out, // [N, 3, N, 3]
        NumericType* mp_out // [P, N, 3]
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

                NumericType dx = x0-x1;
                NumericType dy = y0-y1;
                NumericType dz = z0-z1;

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

                const size_t i_idx = atom_0_idx;
                const size_t j_idx = atom_1_idx;
                const size_t x_dim = 0;
                const size_t y_dim = 1;
                const size_t z_dim = 2;
                const size_t N = n_atoms;

                NumericType d2x = dx*dx;
                NumericType d2y = dy*dy;
                NumericType d2z = dz*dz;

                NumericType dij = timemachine::norm(dx, dy, dz);
                NumericType d2ij = dij*dij;
                NumericType d3ij = d2ij*dij;
                NumericType d5ij = d3ij*d2ij;


                NumericType PREFACTOR_QI_GRAD = scale*ONE_4PI_EPS0*q1/d3ij;
                NumericType PREFACTOR_QJ_GRAD = scale*ONE_4PI_EPS0*q0/d3ij;

                NumericType *mp_out_qi = mp_out + global_param_idxs_[param_idxs_[atom_0_idx]]*n_atoms*3;
                NumericType *mp_out_qj = mp_out + global_param_idxs_[param_idxs_[atom_1_idx]]*n_atoms*3;

                // use symmetry later on
                mp_out_qi[atom_0_idx*3 + 0] += PREFACTOR_QI_GRAD * (-dx);
                mp_out_qi[atom_0_idx*3 + 1] += PREFACTOR_QI_GRAD * (-dy);
                mp_out_qi[atom_0_idx*3 + 2] += PREFACTOR_QI_GRAD * (-dz);
                mp_out_qi[atom_1_idx*3 + 0] += PREFACTOR_QI_GRAD * (dx);
                mp_out_qi[atom_1_idx*3 + 1] += PREFACTOR_QI_GRAD * (dy);
                mp_out_qi[atom_1_idx*3 + 2] += PREFACTOR_QI_GRAD * (dz);

                mp_out_qj[atom_0_idx*3 + 0] += PREFACTOR_QJ_GRAD * (-dx);
                mp_out_qj[atom_0_idx*3 + 1] += PREFACTOR_QJ_GRAD * (-dy);
                mp_out_qj[atom_0_idx*3 + 2] += PREFACTOR_QJ_GRAD * (-dz);
                mp_out_qj[atom_1_idx*3 + 0] += PREFACTOR_QJ_GRAD * (dx);
                mp_out_qj[atom_1_idx*3 + 1] += PREFACTOR_QJ_GRAD * (dy);
                mp_out_qj[atom_1_idx*3 + 2] += PREFACTOR_QJ_GRAD * (dz);


                NumericType inv_d5ij = 1/d5ij;
                NumericType prefactor = scale*ONE_4PI_EPS0*q0*q1*inv_d5ij;

                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*(-d2ij + 3*d2x);
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim] += 3*prefactor*dx*dy;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim] += 3*prefactor*dx*dz;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*(d2ij - 3*d2x);
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim] += -3*prefactor*dx*dy;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim] += -3*prefactor*dx*dz;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim] += 3*prefactor*dx*dy;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*(-d2ij + 3*d2y);
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim] += 3*prefactor*dy*dz;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim] += -3*prefactor*dx*dy;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*(d2ij - 3*d2y);
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim] += -3*prefactor*dy*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim] += 3*prefactor*dx*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim] += 3*prefactor*dy*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*(-d2ij + 3*d2z);
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim] += -3*prefactor*dx*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim] += -3*prefactor*dy*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*(d2ij - 3*d2z);

                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*(d2ij - 3*d2x);
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim] += -3*prefactor*dx*dy;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim] += -3*prefactor*dx*dz;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*(-d2ij + 3*d2x);
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim] += 3*prefactor*dx*dy;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim] += 3*prefactor*dx*dz;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim] += -3*prefactor*dx*dy;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*(d2ij - 3*d2y);
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim] += -3*prefactor*dy*dz;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim] += 3*prefactor*dx*dy;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*(-d2ij + 3*d2y);
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim] += 3*prefactor*dy*dz;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim] += -3*prefactor*dx*dz;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim] += -3*prefactor*dy*dz;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*(d2ij - 3*d2z);
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim] += 3*prefactor*dx*dz;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim] += 3*prefactor*dy*dz;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*(-d2ij + 3*d2z);

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

        OutType dEdx = 24*eps*dx*(sig12rij7*2 - sig6rij4);
        OutType dEdy = 24*eps*dy*(sig12rij7*2 - sig6rij4);
        OutType dEdz = 24*eps*dz*(sig12rij7*2 - sig6rij4);

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
        NumericType* energy_out, // []
        NumericType* grad_out, // [N,3]
        NumericType* hessian_out, // [N, 3, N, 3]
        NumericType* mp_out // [P, N, 3]
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

                // NumericType dx = x0 - x1;
                // NumericType dy = y0 - y1;
                // NumericType dz = z0 - z1;

                // NumericType dij = timemachine::norm(dx, dy, dz);

                const NumericType sig1 = params_[param_idxs_[atom_1_idx*2+0]];
                const NumericType eps1 = params_[param_idxs_[atom_1_idx*2+1]];

                NumericType sig = (sig0 + sig1)/2;
                NumericType eps = sqrt(eps0*eps1);
                // if(dij > 2.5*sig) {
                    // continue;
                // }

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

                const size_t i_idx = atom_0_idx;
                const size_t j_idx = atom_1_idx;
                const size_t x_dim = 0;
                const size_t y_dim = 1;
                const size_t z_dim = 2;
                const size_t N = n_atoms;

                NumericType dx = x0 - x1;
                NumericType dy = y0 - y1;
                NumericType dz = z0 - z1;

                NumericType dij = timemachine::norm(dx, dy, dz);
                NumericType d2ij = dij*dij;
                NumericType d4ij = d2ij*d2ij;
                NumericType d6ij = d4ij*d2ij;
                NumericType d8ij = d4ij*d4ij;
                NumericType d16ij = d8ij*d8ij;
                NumericType inv_d16ij = 1.0/d16ij;

                NumericType sig2 = sig*sig;
                NumericType sig3 = sig2*sig;
                NumericType sig5 = sig3*sig2;
                NumericType sig6 = sig3*sig3;
                NumericType sig11 = sig6*sig3*sig2;
                NumericType sig12 = sig6*sig6;

                NumericType prefactor = scale*eps*sig6;

                NumericType *mp_out_sig0 = mp_out + global_param_idxs_[param_idxs_[atom_0_idx*2+0]]*n_atoms*3;
                NumericType *mp_out_eps0 = mp_out + global_param_idxs_[param_idxs_[atom_0_idx*2+1]]*n_atoms*3;
                NumericType *mp_out_sig1 = mp_out + global_param_idxs_[param_idxs_[atom_1_idx*2+0]]*n_atoms*3;
                NumericType *mp_out_eps1 = mp_out + global_param_idxs_[param_idxs_[atom_1_idx*2+1]]*n_atoms*3;


                NumericType rij = dx*dx + dy*dy + dz*dz;
                NumericType rij3 = rij * rij * rij;
                NumericType rij4 = rij3 * rij;
                NumericType rij7 = rij4 * rij3;

                // (ytz): 99 % sure this loses precision so we need to refactor
                NumericType sig12rij7 = sig12/rij7;
                NumericType sig11rij7 = sig11/rij7;
                NumericType sig6rij4 = sig6/rij4;
                NumericType sig5rij4 = sig5/rij4;

                mp_out_eps0[atom_0_idx*3 + 0] += -scale*12*(eps1/eps)*dx*(sig12rij7*2 - sig6rij4);
                mp_out_eps0[atom_0_idx*3 + 1] += -scale*12*(eps1/eps)*dy*(sig12rij7*2 - sig6rij4);
                mp_out_eps0[atom_0_idx*3 + 2] += -scale*12*(eps1/eps)*dz*(sig12rij7*2 - sig6rij4);

                mp_out_eps1[atom_0_idx*3 + 0] += -scale*12*(eps0/eps)*dx*(sig12rij7*2 - sig6rij4);
                mp_out_eps1[atom_0_idx*3 + 1] += -scale*12*(eps0/eps)*dy*(sig12rij7*2 - sig6rij4);
                mp_out_eps1[atom_0_idx*3 + 2] += -scale*12*(eps0/eps)*dz*(sig12rij7*2 - sig6rij4);

                mp_out_eps0[atom_1_idx*3 + 0] +=  scale*12*(eps1/eps)*dx*(sig12rij7*2 - sig6rij4);
                mp_out_eps0[atom_1_idx*3 + 1] +=  scale*12*(eps1/eps)*dy*(sig12rij7*2 - sig6rij4);
                mp_out_eps0[atom_1_idx*3 + 2] +=  scale*12*(eps1/eps)*dz*(sig12rij7*2 - sig6rij4);

                mp_out_eps1[atom_1_idx*3 + 0] +=  scale*12*(eps0/eps)*dx*(sig12rij7*2 - sig6rij4);
                mp_out_eps1[atom_1_idx*3 + 1] +=  scale*12*(eps0/eps)*dy*(sig12rij7*2 - sig6rij4);
                mp_out_eps1[atom_1_idx*3 + 2] +=  scale*12*(eps0/eps)*dz*(sig12rij7*2 - sig6rij4);

                mp_out_sig0[atom_0_idx*3 + 0] += -scale*24*eps*dx*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig0[atom_0_idx*3 + 1] += -scale*24*eps*dy*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig0[atom_0_idx*3 + 2] += -scale*24*eps*dz*(12*sig11rij7 - 3*sig5rij4);

                mp_out_sig1[atom_0_idx*3 + 0] += -scale*24*eps*dx*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig1[atom_0_idx*3 + 1] += -scale*24*eps*dy*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig1[atom_0_idx*3 + 2] += -scale*24*eps*dz*(12*sig11rij7 - 3*sig5rij4);

                mp_out_sig0[atom_1_idx*3 + 0] +=  scale*24*eps*dx*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig0[atom_1_idx*3 + 1] +=  scale*24*eps*dy*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig0[atom_1_idx*3 + 2] +=  scale*24*eps*dz*(12*sig11rij7 - 3*sig5rij4);

                mp_out_sig1[atom_1_idx*3 + 0] +=  scale*24*eps*dx*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig1[atom_1_idx*3 + 1] +=  scale*24*eps*dy*(12*sig11rij7 - 3*sig5rij4);
                mp_out_sig1[atom_1_idx*3 + 2] +=  scale*24*eps*dz*(12*sig11rij7 - 3*sig5rij4);

                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*-96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*-96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*-24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*-96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*-96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*-24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*-96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*-96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*-24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*-24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*-96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*-96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*-24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*-96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*-96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*-24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*-96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*-96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;

                // for(size_t j=0; j < 6; j++) {

                //     std::array<std::complex<NumericType>, 6> cxs = timemachine::convert_to_complex(xs);
                //     cxs[j] = std::complex<NumericType>(cxs[j].real(), step);
                //     std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<std::complex<NumericType>, NumericType, std::complex<NumericType> >(cxs, params);
                //     std::array<NumericType, 6> ddxs;
                //     for(int k=0; k < 6; k++) {
                //         ddxs[k] = scale*dcxs[k].imag() / step;
                //         hessian_out[indices[j]*n_atoms*3 + indices[k]] += scale*(dcxs[k].imag() / step);
                //     }
                // }

            }
        }
    }
};

} // namespace timemachine