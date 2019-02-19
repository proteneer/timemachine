#pragma once
#include <array>
#include <iostream>
#include <complex>
#include <vector>
#include "utils.hpp"

namespace timemachine {

float complex_atan2(float z1, float z2) {
    return std::atan2(z1, z2);
}


double complex_atan2(double z1, double z2) {
    return std::atan2(z1, z2);
}

std::complex<double> complex_atan2(std::complex<double> z1, std::complex<double> z2) {
    double real_part = std::atan2(z1.real(),z2.real());
    double imag_part = (z2.real()*z1.imag()-z1.real()*z2.imag())/(z1.real()*z1.real()+z2.real()*z2.real());
    return std::complex<double>(real_part, imag_part);
}

std::complex<float> complex_atan2(std::complex<float> z1, std::complex<float> z2) {
    float real_part = std::atan2(z1.real(),z2.real());
    float imag_part = (z2.real()*z1.imag()-z1.real()*z2.imag())/(z1.real()*z1.real()+z2.real()*z2.real());
    return std::complex<float>(real_part, imag_part);
}

template <typename NumericType>
class PeriodicTorsion {

private:

    const std::vector<NumericType> params_;
    const std::vector<size_t> global_param_idxs_;
    const std::vector<size_t> param_idxs_;
    const std::vector<size_t> tors_idxs_;


public:

    PeriodicTorsion(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<size_t> tors_idxs
    ) : params_(params),
        global_param_idxs_(global_param_idxs),
        param_idxs_(param_idxs),
        tors_idxs_(tors_idxs) {};

    size_t numTorsions() const {
        return tors_idxs_.size()/4;
    }

    NumericType ixn_energy(
        const std::array<NumericType, 12> &xs,
        const std::array<NumericType, 3> &params) const {
        NumericType x0 = xs[0];
        NumericType y0 = xs[1];
        NumericType z0 = xs[2];
        NumericType x1 = xs[3];
        NumericType y1 = xs[4];
        NumericType z1 = xs[5];
        NumericType x2 = xs[6];
        NumericType y2 = xs[7];
        NumericType z2 = xs[8];
        NumericType x3 = xs[9];
        NumericType y3 = xs[10];
        NumericType z3 = xs[11];
        NumericType k = params[0];
        NumericType phase = params[1];
        NumericType period = params[2];

        NumericType rij_x = x0 - x1;
        NumericType rij_y = y0 - y1;
        NumericType rij_z = z0 - z1;

        NumericType rkj_x = x2 - x1;
        NumericType rkj_y = y2 - y1;
        NumericType rkj_z = z2 - z1;

        NumericType rkl_x = x2 - x3;
        NumericType rkl_y = y2 - y3;
        NumericType rkl_z = z2 - z3;

        NumericType n1_x, n1_y, n1_z, n2_x, n2_y, n2_z;

        timemachine::cross_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z, n1_x, n1_y, n1_z);
        timemachine::cross_product(rkj_x, rkj_y, rkj_z, rkl_x, rkl_y, rkl_z, n2_x, n2_y, n2_z);

        // NumericType lhs = timemachine::norm(n1_x, n1_y, n1_z);
        // NumericType rhs = timemachine::norm(n2_x, n2_y, n2_z);
        // NumericType bot = lhs * rhs;

        NumericType n3_x, n3_y, n3_z;

        timemachine::cross_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z, n3_x, n3_y, n3_z);

        NumericType rkj_n = timemachine::norm(rkj_x, rkj_y, rkj_z);
        rkj_x /= rkj_n;
        rkj_y /= rkj_n;
        rkj_z /= rkj_n;

        NumericType y = timemachine::dot_product(n3_x, n3_y, n3_z, rkj_x, rkj_y, rkj_z);
        NumericType x = timemachine::dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z);
        NumericType angle = std::atan2(y, x);

        // (ytz): The sign version is commented out due to yutong not being able to figure out
        // the correct complex step derivative.
        // NumericType sign_angle = sign(timemachine::dot_product(rkj_x, rkj_y, rkj_z, n3_x, n3_y, n3_z));
        // NumericType angle = sign_angle*acos(timemachine::dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z)/bot);
        return k*(1+cos(period*angle - phase));

    }


    template<
        typename CoordType,
        typename ParamType,
        typename OutType>
    std::array<OutType, 12> ixn_gradient(
        const std::array<CoordType, 12> &xs,
        const std::array<ParamType, 3> &params) const {
        CoordType x0 = xs[0];
        CoordType y0 = xs[1];
        CoordType z0 = xs[2];
        CoordType x1 = xs[3];
        CoordType y1 = xs[4];
        CoordType z1 = xs[5];
        CoordType x2 = xs[6];
        CoordType y2 = xs[7];
        CoordType z2 = xs[8];
        CoordType x3 = xs[9];
        CoordType y3 = xs[10];
        CoordType z3 = xs[11];
        ParamType k = params[0];
        ParamType phase = params[1];
        ParamType period = params[2];

        CoordType rij_x = x0 - x1;
        CoordType rij_y = y0 - y1;
        CoordType rij_z = z0 - z1;

        CoordType rkj_x = x2 - x1;
        CoordType rkj_y = y2 - y1;
        CoordType rkj_z = z2 - z1;

        CoordType rkj_norm = timemachine::norm(rkj_x, rkj_y, rkj_z);
        CoordType rkj_norm_square = timemachine::dot_product(rkj_x, rkj_y, rkj_z, rkj_x, rkj_y, rkj_z);

        CoordType rkl_x = x2 - x3;
        CoordType rkl_y = y2 - y3;
        CoordType rkl_z = z2 - z3;

        CoordType n1_x, n1_y, n1_z, n2_x, n2_y, n2_z;

        timemachine::cross_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z, n1_x, n1_y, n1_z);
        timemachine::cross_product(rkj_x, rkj_y, rkj_z, rkl_x, rkl_y, rkl_z, n2_x, n2_y, n2_z);

        // CoordType n1_norm = timemachine::norm(n1_x, n1_y, n1_z);
        CoordType n1_norm_square = timemachine::dot_product(n1_x, n1_y, n1_z, n1_x, n1_y, n1_z);
        // CoordType n2_norm = timemachine::norm(n2_x, n2_y, n2_z);
        CoordType n2_norm_square = timemachine::dot_product(n2_x, n2_y, n2_z, n2_x, n2_y, n2_z);
        // CoordType bot = n1_norm * n2_norm;

        CoordType n3_x, n3_y, n3_z;

        timemachine::cross_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z, n3_x, n3_y, n3_z);

        CoordType dangle_dR0_x = rkj_norm/(n1_norm_square) * n1_x;
        CoordType dangle_dR0_y = rkj_norm/(n1_norm_square) * n1_y;
        CoordType dangle_dR0_z = rkj_norm/(n1_norm_square) * n1_z; 

        CoordType dangle_dR3_x = -rkj_norm/(n2_norm_square) * n2_x;
        CoordType dangle_dR3_y = -rkj_norm/(n2_norm_square) * n2_y;
        CoordType dangle_dR3_z = -rkj_norm/(n2_norm_square) * n2_z; 

        CoordType dangle_dR1_x = (timemachine::dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_x - dangle_dR3_x*timemachine::dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
        CoordType dangle_dR1_y = (timemachine::dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_y - dangle_dR3_y*timemachine::dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
        CoordType dangle_dR1_z = (timemachine::dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_z - dangle_dR3_z*timemachine::dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);

        CoordType dangle_dR2_x = (timemachine::dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_x - dangle_dR0_x*timemachine::dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
        CoordType dangle_dR2_y = (timemachine::dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_y - dangle_dR0_y*timemachine::dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
        CoordType dangle_dR2_z = (timemachine::dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_z - dangle_dR0_z*timemachine::dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);

        CoordType rkj_n = timemachine::norm(rkj_x, rkj_y, rkj_z);
        rkj_x /= rkj_n;
        rkj_y /= rkj_n;
        rkj_z /= rkj_n;

        CoordType y = timemachine::dot_product(n3_x, n3_y, n3_z, rkj_x, rkj_y, rkj_z);
        CoordType x = timemachine::dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z);
        CoordType angle = complex_atan2(y, x);

        // CoordType sign_angle = sign(timemachine::dot_product(rkj_x, rkj_y, rkj_z, n3_x, n3_y, n3_z));
        // CoordType angle = sign_angle*acos(timemachine::dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z)/(bot));
        OutType prefactor = -k*sin(period*angle - phase)*period;

        std::array<OutType, 12> grads;

        grads[0*3+0] = dangle_dR0_x * prefactor;
        grads[0*3+1] = dangle_dR0_y * prefactor;
        grads[0*3+2] = dangle_dR0_z * prefactor;

        grads[1*3+0] = dangle_dR1_x * prefactor;
        grads[1*3+1] = dangle_dR1_y * prefactor;
        grads[1*3+2] = dangle_dR1_z * prefactor;

        grads[2*3+0] = dangle_dR2_x * prefactor;
        grads[2*3+1] = dangle_dR2_y * prefactor;
        grads[2*3+2] = dangle_dR2_z * prefactor;

        grads[3*3+0] = dangle_dR3_x * prefactor;
        grads[3*3+1] = dangle_dR3_y * prefactor;
        grads[3*3+2] = dangle_dR3_z * prefactor;

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

        for(size_t i=0; i < numTorsions(); i++) {
            size_t atom_0_idx = tors_idxs_[i*4+0];
            size_t atom_1_idx = tors_idxs_[i*4+1];
            size_t atom_2_idx = tors_idxs_[i*4+2];
            size_t atom_3_idx = tors_idxs_[i*4+3];

            size_t p_k_idx = param_idxs_[i*3+0];
            size_t p_phase_idx = param_idxs_[i*3+1];
            size_t p_period_idx = param_idxs_[i*3+2];
            NumericType k = params_[p_k_idx];
            NumericType period = params_[p_period_idx];
            NumericType phase = params_[p_phase_idx];

            const NumericType x0 = coords[atom_0_idx*3+0];
            const NumericType y0 = coords[atom_0_idx*3+1];
            const NumericType z0 = coords[atom_0_idx*3+2];
            const NumericType x1 = coords[atom_1_idx*3+0];
            const NumericType y1 = coords[atom_1_idx*3+1];
            const NumericType z1 = coords[atom_1_idx*3+2];
            const NumericType x2 = coords[atom_2_idx*3+0];
            const NumericType y2 = coords[atom_2_idx*3+1];
            const NumericType z2 = coords[atom_2_idx*3+2];
            const NumericType x3 = coords[atom_3_idx*3+0];
            const NumericType y3 = coords[atom_3_idx*3+1];
            const NumericType z3 = coords[atom_3_idx*3+2];

            std::array<NumericType, 12> xs({x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3});
            std::array<NumericType, 3> params({k, phase, period});
            *energy_out += ixn_energy(xs, params);

            std::array<NumericType, 12> grads = ixn_gradient<NumericType, NumericType, NumericType>(xs, params);

            grad_out[atom_0_idx*3 + 0] += grads[0];
            grad_out[atom_0_idx*3 + 1] += grads[1];
            grad_out[atom_0_idx*3 + 2] += grads[2];
            grad_out[atom_1_idx*3 + 0] += grads[3];
            grad_out[atom_1_idx*3 + 1] += grads[4];
            grad_out[atom_1_idx*3 + 2] += grads[5];
            grad_out[atom_2_idx*3 + 0] += grads[6];
            grad_out[atom_2_idx*3 + 1] += grads[7];
            grad_out[atom_2_idx*3 + 2] += grads[8];
            grad_out[atom_3_idx*3 + 0] += grads[9];
            grad_out[atom_3_idx*3 + 1] += grads[10];
            grad_out[atom_3_idx*3 + 2] += grads[11];

            NumericType step = 1e-50;

            // compute mixed partials, loop over all the parameters
            // (ytz) TODO: unroll this
            for(size_t j=0; j < 3; j++) {
                std::array<std::complex<NumericType>, 3> cparams = timemachine::convert_to_complex(params);
                cparams[j] = std::complex<NumericType>(cparams[j].real(), step);

                std::array<std::complex<NumericType>, 12> dcxs = ixn_gradient<NumericType, std::complex<NumericType>, std::complex<NumericType> >(xs, cparams);

                std::array<NumericType, 12> ddxs;
                for(int k=0; k < 12; k++) {
                    ddxs[k] = dcxs[k].imag() / step;
                }
                size_t p_idx = global_param_idxs_[param_idxs_[i*3+j]];
                mp_out[p_idx*n_atoms*3 + atom_0_idx*3 + 0] += ddxs[0];
                mp_out[p_idx*n_atoms*3 + atom_0_idx*3 + 1] += ddxs[1];
                mp_out[p_idx*n_atoms*3 + atom_0_idx*3 + 2] += ddxs[2];
                mp_out[p_idx*n_atoms*3 + atom_1_idx*3 + 0] += ddxs[3];
                mp_out[p_idx*n_atoms*3 + atom_1_idx*3 + 1] += ddxs[4];
                mp_out[p_idx*n_atoms*3 + atom_1_idx*3 + 2] += ddxs[5];
                mp_out[p_idx*n_atoms*3 + atom_2_idx*3 + 0] += ddxs[6];
                mp_out[p_idx*n_atoms*3 + atom_2_idx*3 + 1] += ddxs[7];
                mp_out[p_idx*n_atoms*3 + atom_2_idx*3 + 2] += ddxs[8];
                mp_out[p_idx*n_atoms*3 + atom_3_idx*3 + 0] += ddxs[9];
                mp_out[p_idx*n_atoms*3 + atom_3_idx*3 + 1] += ddxs[10];
                mp_out[p_idx*n_atoms*3 + atom_3_idx*3 + 2] += ddxs[11];
            }

            // compute hessian vector product, looping over all atoms and parameters
            std::array<size_t, 12> indices({
                atom_0_idx*3+0,
                atom_0_idx*3+1,
                atom_0_idx*3+2,
                atom_1_idx*3+0,
                atom_1_idx*3+1,
                atom_1_idx*3+2,
                atom_2_idx*3+0,
                atom_2_idx*3+1,
                atom_2_idx*3+2,
                atom_3_idx*3+0,
                atom_3_idx*3+1,
                atom_3_idx*3+2
            });

            for(size_t j=0; j < 12; j++) {

                std::array<std::complex<NumericType>, 12> cxs = timemachine::convert_to_complex(xs);
                cxs[j] = std::complex<NumericType>(cxs[j].real(), step);
                std::array<std::complex<NumericType>, 12> dcxs = ixn_gradient<std::complex<NumericType>, NumericType, std::complex<NumericType> >(cxs, params);
                // std::array<NumericType, 12> ddxs;
                for(int k=0; k < 12; k++) {
                    // ddxs[k] = dcxs[k].imag() / step;
                    hessian_out[indices[j]*n_atoms*3 + indices[k]] += dcxs[k].imag() / step;
                }

                // this loops over *all* params, not just the two params specific to an angle.     
                // for(size_t p=0; p < n_params; p++) {
                //     auto dx0 = ddxs[0] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 0] + ddxs[3] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 0] + ddxs[6] * dxdp[p*n_atoms*3 + atom_2_idx*3 + 0] + ddxs[9] * dxdp[p*n_atoms*3 + atom_3_idx*3 + 0];
                //     auto dx1 = ddxs[1] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 1] + ddxs[4] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 1] + ddxs[7] * dxdp[p*n_atoms*3 + atom_2_idx*3 + 1] + ddxs[10] * dxdp[p*n_atoms*3 + atom_3_idx*3 + 1];
                //     auto dx2 = ddxs[2] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 2] + ddxs[5] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 2] + ddxs[8] * dxdp[p*n_atoms*3 + atom_2_idx*3 + 2] + ddxs[11] * dxdp[p*n_atoms*3 + atom_3_idx*3 + 2];
                //     total_out[p*n_atoms*3 + indices[j]] += dx0+dx1+dx2;
                // }
            }
        }
    }
};


template <typename NumericType>
class HarmonicAngle {

private:

    const std::vector<NumericType> params_;
    const std::vector<size_t> global_param_idxs_;
    const std::vector<size_t> param_idxs_;
    const std::vector<size_t> angle_idxs_;
    bool cos_angles_;

public:

    HarmonicAngle(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<size_t> angle_idxs,
        bool cos_angles=true
    ) : params_(params), 
        global_param_idxs_(global_param_idxs),
        param_idxs_(param_idxs),
        angle_idxs_(angle_idxs),
        cos_angles_(cos_angles) {};

    size_t numAngles() const {
        return angle_idxs_.size()/3;
    }

    NumericType ixn_energy(
        const std::array<NumericType, 9> &xs,
        const std::array<NumericType, 2> &params) const {
        NumericType x0 = xs[0];
        NumericType y0 = xs[1];
        NumericType z0 = xs[2];
        NumericType x1 = xs[3];
        NumericType y1 = xs[4];
        NumericType z1 = xs[5];
        NumericType x2 = xs[6];
        NumericType y2 = xs[7];
        NumericType z2 = xs[8];
        NumericType ka = params[0];
        NumericType a0 = params[1];

        NumericType vij_x = x1 - x0;
        NumericType vij_y = y1 - y0;
        NumericType vij_z = z1 - z0;

        NumericType vjk_x = x1 - x2;
        NumericType vjk_y = y1 - y2;
        NumericType vjk_z = z1 - z2;

        NumericType nij = timemachine::norm(vij_x, vij_y, vij_z);
        NumericType njk = timemachine::norm(vjk_x, vjk_y, vjk_z);
        NumericType top = timemachine::dot_product(vij_x, vij_y, vij_z, vjk_x, vjk_y, vjk_z);
        NumericType ca = top/(nij*njk);

        NumericType delta;
        if(!cos_angles_) {  
            delta = acos(ca) - a0;
        } else {
            delta = ca - cos(a0);
        }
        return ka/2*(delta*delta);

    }

    template<
        typename CoordType,
        typename ParamType,
        typename OutType>
    std::array<OutType, 9> ixn_gradient(
        const std::array<CoordType, 9> &xs,
        const std::array<ParamType, 2> &params) const {
        CoordType x0 = xs[0];
        CoordType y0 = xs[1];
        CoordType z0 = xs[2];
        CoordType x1 = xs[3];
        CoordType y1 = xs[4];
        CoordType z1 = xs[5];
        CoordType x2 = xs[6];
        CoordType y2 = xs[7];
        CoordType z2 = xs[8];
        ParamType ka = params[0];
        ParamType a0 = params[1];

        CoordType vij_x = x1 - x0;
        CoordType vij_y = y1 - y0;
        CoordType vij_z = z1 - z0;

        CoordType vjk_x = x1 - x2;
        CoordType vjk_y = y1 - y2;
        CoordType vjk_z = z1 - z2;

        CoordType nij = timemachine::norm(vij_x, vij_y, vij_z);
        CoordType njk = timemachine::norm(vjk_x, vjk_y, vjk_z);

        // can be optimized
        CoordType n3ij = nij*nij*nij;
        CoordType n3jk = njk*njk*njk;
        CoordType top = timemachine::dot_product(vij_x, vij_y, vij_z, vjk_x, vjk_y, vjk_z);


        std::array<OutType, 9> grads; // uninitialized

        grads[0*3+0] = 0.5*ka*((top)/(nij*njk) - cos(a0))*(2.0*(-x0 + x1)*(top)/(n3ij*njk) + 2.0*(-x1 + x2)/(nij*njk)) ;
        grads[1*3+0] = 0.5*ka*((top)/(nij*njk) - cos(a0))*(2.0*(x0 - x1)*(top)/(n3ij*njk) + 2.0*(-x1 + x2)*(top)/(nij*n3jk) + 2.0*(-x0 + 2.0*x1 - x2)/(nij*njk)) ;
        grads[2*3+0] = 0.5*ka*(2.0*(x0 - x1)/(nij*njk) + 2.0*(x1 - x2)*(top)/(nij*n3jk))*((top)/(nij*njk) - cos(a0)) ;
        grads[0*3+1] = 0.5*ka*((top)/(nij*njk) - cos(a0))*(2.0*(-y0 + y1)*(top)/(n3ij*njk) + 2.0*(-y1 + y2)/(nij*njk)) ;
        grads[1*3+1] = 0.5*ka*((top)/(nij*njk) - cos(a0))*(2.0*(y0 - y1)*(top)/(n3ij*njk) + 2.0*(-y1 + y2)*(top)/(nij*n3jk) + 2.0*(-y0 + 2.0*y1 - y2)/(nij*njk)) ;
        grads[2*3+1] = 0.5*ka*(2.0*(y0 - y1)/(nij*njk) + 2.0*(y1 - y2)*(top)/(nij*n3jk))*((top)/(nij*njk) - cos(a0)) ;
        grads[0*3+2] = 0.5*ka*((top)/(nij*njk) - cos(a0))*(2.0*(-z0 + z1)*(top)/(n3ij*njk) + 2.0*(-z1 + z2)/(nij*njk)) ;
        grads[1*3+2] = 0.5*ka*((top)/(nij*njk) - cos(a0))*(2.0*(z0 - z1)*(top)/(n3ij*njk) + 2.0*(-z1 + z2)*(top)/(nij*n3jk) + 2.0*(-z0 + 2.0*z1 - z2)/(nij*njk)) ;
        grads[2*3+2] = 0.5*ka*(2.0*(z0 - z1)/(nij*njk) + 2.0*(z1 - z2)*(top)/(nij*n3jk))*((top)/(nij*njk) - cos(a0)) ;

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

        for(size_t i=0; i < numAngles(); i++) {

            size_t atom_0_idx = angle_idxs_[i*3+0];
            size_t atom_1_idx = angle_idxs_[i*3+1];
            size_t atom_2_idx = angle_idxs_[i*3+2];
            size_t p_k_idx = param_idxs_[i*2+0];
            size_t p_a_idx = param_idxs_[i*2+1];
            NumericType kb = params_[p_k_idx];
            NumericType a0 = params_[p_a_idx];

            if(atom_0_idx == atom_1_idx) {
                throw std::runtime_error("self-bond detected.");
            }

            const NumericType x0 = coords[atom_0_idx*3+0];
            const NumericType y0 = coords[atom_0_idx*3+1];
            const NumericType z0 = coords[atom_0_idx*3+2];
            const NumericType x1 = coords[atom_1_idx*3+0];
            const NumericType y1 = coords[atom_1_idx*3+1];
            const NumericType z1 = coords[atom_1_idx*3+2];
            const NumericType x2 = coords[atom_2_idx*3+0];
            const NumericType y2 = coords[atom_2_idx*3+1];
            const NumericType z2 = coords[atom_2_idx*3+2];

            std::array<NumericType, 9> xs({x0, y0, z0, x1, y1, z1, x2, y2, z2});
            std::array<NumericType, 2> params({kb, a0});

            *energy_out += ixn_energy(xs, params);

            // (ytz): very important that we initialize the array to zero.
            std::array<NumericType, 9> grads = ixn_gradient<NumericType, NumericType, NumericType>(xs, params);

            grad_out[atom_0_idx*3 + 0] += grads[0];
            grad_out[atom_0_idx*3 + 1] += grads[1];
            grad_out[atom_0_idx*3 + 2] += grads[2];
            grad_out[atom_1_idx*3 + 0] += grads[3];
            grad_out[atom_1_idx*3 + 1] += grads[4];
            grad_out[atom_1_idx*3 + 2] += grads[5];
            grad_out[atom_2_idx*3 + 0] += grads[6];
            grad_out[atom_2_idx*3 + 1] += grads[7];
            grad_out[atom_2_idx*3 + 2] += grads[8];

            NumericType step = 1e-100;

            // compute mixed partials, loop over all the parameters
            // (ytz) TODO: unroll this
            for(size_t j=0; j < 2; j++) {
                std::array<std::complex<NumericType>, 2> cparams = timemachine::convert_to_complex(params);
                cparams[j] = std::complex<NumericType>(cparams[j].real(), step);

                std::array<std::complex<NumericType>, 9> dcxs = ixn_gradient<NumericType, std::complex<NumericType>, std::complex<NumericType> >(xs, cparams);

                std::array<NumericType, 9> ddxs;
                for(int k=0; k < 9; k++) {
                    ddxs[k] = dcxs[k].imag() / step;
                }
                size_t p_idx = global_param_idxs_[param_idxs_[i*2+j]];
                mp_out[p_idx*n_atoms*3 + atom_0_idx*3 + 0] += ddxs[0];
                mp_out[p_idx*n_atoms*3 + atom_0_idx*3 + 1] += ddxs[1];
                mp_out[p_idx*n_atoms*3 + atom_0_idx*3 + 2] += ddxs[2];
                mp_out[p_idx*n_atoms*3 + atom_1_idx*3 + 0] += ddxs[3];
                mp_out[p_idx*n_atoms*3 + atom_1_idx*3 + 1] += ddxs[4];
                mp_out[p_idx*n_atoms*3 + atom_1_idx*3 + 2] += ddxs[5];
                mp_out[p_idx*n_atoms*3 + atom_2_idx*3 + 0] += ddxs[6];
                mp_out[p_idx*n_atoms*3 + atom_2_idx*3 + 1] += ddxs[7];
                mp_out[p_idx*n_atoms*3 + atom_2_idx*3 + 2] += ddxs[8];
            }

            // compute hessian vector product, looping over all atoms and parameters
            std::array<size_t, 9> indices({
                atom_0_idx*3+0,
                atom_0_idx*3+1,
                atom_0_idx*3+2,
                atom_1_idx*3+0,
                atom_1_idx*3+1,
                atom_1_idx*3+2,
                atom_2_idx*3+0,
                atom_2_idx*3+1,
                atom_2_idx*3+2
            });

            for(size_t j=0; j < 9; j++) {

                std::array<std::complex<NumericType>, 9> cxs = timemachine::convert_to_complex(xs);
                cxs[j] = std::complex<NumericType>(cxs[j].real(), step);
                std::array<std::complex<NumericType>, 9> dcxs = ixn_gradient<std::complex<NumericType>, NumericType, std::complex<NumericType> >(cxs, params);
                // std::array<NumericType, 9> ddxs;
                for(int k=0; k < 9; k++) {
                    // ddxs[k] = dcxs[k].imag() / step;
                    hessian_out[indices[j]*n_atoms*3 + indices[k]] += dcxs[k].imag() / step;
                }

                // this loops over *all* params, not just the two params specific to an angle.     
                // for(size_t p=0; p < n_params; p++) {
                //     auto dx0 = ddxs[0] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 0] + ddxs[3] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 0] + ddxs[6] * dxdp[p*n_atoms*3 + atom_2_idx*3 + 0];
                //     auto dx1 = ddxs[1] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 1] + ddxs[4] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 1] + ddxs[7] * dxdp[p*n_atoms*3 + atom_2_idx*3 + 1];
                //     auto dx2 = ddxs[2] * dxdp[p*n_atoms*3 + atom_0_idx*3 + 2] + ddxs[5] * dxdp[p*n_atoms*3 + atom_1_idx*3 + 2] + ddxs[8] * dxdp[p*n_atoms*3 + atom_2_idx*3 + 2];
                //     total_out[p*n_atoms*3 + indices[j]] += dx0+dx1+dx2;
                // }
            }
        }
    }
};

template <typename NumericType>
class HarmonicBond {

private:

    const std::vector<NumericType> params_;
    const std::vector<size_t> global_param_idxs_; // scatter_idxs
    const std::vector<size_t> param_idxs_;
    const std::vector<size_t> bond_idxs_;

public:

    HarmonicBond(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<size_t> bond_idxs
    ) : params_(params),
        global_param_idxs_(global_param_idxs),
        param_idxs_(param_idxs),
        bond_idxs_(bond_idxs) {};


    NumericType ixn_energy(
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
    std::array<OutType, 6> ixn_gradient(
        const std::array<CoordType, 6> &xs,
        const std::array<ParamType, 2> &params) const {
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

        std::array<OutType, 6> grads; 

        grads[0] = kb*db*dx/dij;
        grads[1] = kb*db*dy/dij;
        grads[2] = kb*db*dz/dij;
        grads[3] = -grads[0];
        grads[4] = -grads[1];
        grads[5] = -grads[2];

        return grads;
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
        NumericType* energy_out, // []
        NumericType* grad_out, // [N,3]
        NumericType* hessian_out, // [N, 3, N, 3]
        NumericType* mp_out // [P, N, 3]
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

            *energy_out += ixn_energy(xs, params);

            // (ytz): very important that we initialize the array to zero.
            std::array<NumericType, 6> dxs = ixn_gradient<NumericType, NumericType, NumericType>(xs, params);
            grad_out[s_atom_idx*3 + 0] += dxs[0];
            grad_out[s_atom_idx*3 + 1] += dxs[1];
            grad_out[s_atom_idx*3 + 2] += dxs[2];
            grad_out[e_atom_idx*3 + 0] += dxs[3];
            grad_out[e_atom_idx*3 + 1] += dxs[4];
            grad_out[e_atom_idx*3 + 2] += dxs[5];

            NumericType step = 1e-50;

            // compute mixed partials, loop over all the parameters
            // (ytz) TODO: unroll this
            for(size_t j=0; j < 2; j++) {
                std::array<std::complex<NumericType>, 2> cparams = timemachine::convert_to_complex(params);
                cparams[j] = std::complex<NumericType>(cparams[j].real(), step);

                // (ytz): very important that we initialize the array to zero.
                std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<NumericType, std::complex<NumericType>, std::complex<NumericType> >(xs, cparams);

                std::array<NumericType, 6> ddxs;
                for(int k=0; k < 6; k++) {
                    ddxs[k] = dcxs[k].imag() / step;
                }
                size_t p_idx = global_param_idxs_[param_idxs_[i*2+j]];
                mp_out[p_idx*n_atoms*3 + s_atom_idx*3 + 0] += ddxs[0];
                mp_out[p_idx*n_atoms*3 + s_atom_idx*3 + 1] += ddxs[1];
                mp_out[p_idx*n_atoms*3 + s_atom_idx*3 + 2] += ddxs[2];
                mp_out[p_idx*n_atoms*3 + e_atom_idx*3 + 0] += ddxs[3];
                mp_out[p_idx*n_atoms*3 + e_atom_idx*3 + 1] += ddxs[4];
                mp_out[p_idx*n_atoms*3 + e_atom_idx*3 + 2] += ddxs[5];
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

                std::array<std::complex<NumericType>, 6> dcxs = ixn_gradient<std::complex<NumericType>, NumericType, std::complex<NumericType> >(cxs, params);
                // std::array<NumericType, 6> ddxs;
                for(int k=0; k < 6; k++) {
                    // ddxs[k] = dcxs[k].imag() / step;
                    hessian_out[indices[j]*n_atoms*3 + indices[k]] += dcxs[k].imag() / step;
                }

                // this loops over *all* params, not just the two params specific to a bond.
                // for(size_t p=0; p < n_params; p++) {
                //     auto dx0 = ddxs[0] * dxdp[p*n_atoms*3 + s_atom_idx*3 + 0] + ddxs[3] * dxdp[p*n_atoms*3 + e_atom_idx*3 + 0];
                //     auto dx1 = ddxs[1] * dxdp[p*n_atoms*3 + s_atom_idx*3 + 1] + ddxs[4] * dxdp[p*n_atoms*3 + e_atom_idx*3 + 1];
                //     auto dx2 = ddxs[2] * dxdp[p*n_atoms*3 + s_atom_idx*3 + 2] + ddxs[5] * dxdp[p*n_atoms*3 + e_atom_idx*3 + 2];
                //     total_out[p*n_atoms*3 + indices[j]] += dx0+dx1+dx2;
                // }

            }
        }

        // this loops over *all* params, not just the two params specific to a bond.
        // this is even slower!
        // for(size_t p=0; p < n_params; p++) {
        //     std::cout << "explicit HvP: " << p << std::endl;
        //     const NumericType *p_block = dxdp + p*n_atoms*3;
        //     for(size_t row=0; row < n_atoms*3; row++) {
        //         NumericType accum = 0;
        //         NumericType *row_base = &hessians[0] + row*n_atoms*3;
        //         for(size_t col=0; col < n_atoms*3; col++) {
        //             accum += row_base[col] * p_block[col];
        //         }
        //         total_out[p*n_atoms*3 + row] += accum;
        //     }
        // }

    }
};

}