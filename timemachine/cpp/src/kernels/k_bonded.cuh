#include "surreal.cuh"
#include "../fixed_point.hpp"

/*
Bond terms will only ever use the first three dimensions.
*/

#define MAXDIM 3

template<typename RealType>
void __global__ k_harmonic_bond_inference(
    const int B,     // number of bonds
    const double *coords,  // [n, 3]
    const double *params,  // [p, 2]
    const int *bond_idxs,    // [b, 2]
    unsigned long long *grad_coords,
    double *d_du_dp,
    double *energy) {

    const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= B) {
        return;
    }

    int src_idx = bond_idxs[b_idx*2+0];
    int dst_idx = bond_idxs[b_idx*2+1];

    RealType dx[3];
    RealType d2ij = 0; // initialize your summed variables!
    for(int d=0; d < 3; d++) {
        RealType delta = coords[src_idx*3+d] - coords[dst_idx*3+d];
        dx[d] = delta;
        d2ij += delta*delta;
    }


    RealType kb = params[b_idx*2+0];
    RealType b0 = params[b_idx*2+1];

    if(b0 != 0) {

        RealType dij = sqrt(d2ij);
        RealType db = dij - b0;

        for(int d=0; d < 3; d++) {
            grad_delta = kb*db*dx[d]/dij;
            atomicAdd(grad_coords + src_idx*3 + d, static_cast<unsigned long long>((long long) (grad_delta*FIXED_EXPONENT)));
            atomicAdd(grad_coords + dst_idx*3 + d, static_cast<unsigned long long>((long long) (-grad_delta*FIXED_EXPONENT)));
        }

    } else{

        for(int d=0; d < 3; d++) {
            grad_delta = kb*dx[d];
            atomicAdd(grad_coords + src_idx*3 + d, static_cast<unsigned long long>((long long) (grad_delta*FIXED_EXPONENT)));
            atomicAdd(grad_coords + dst_idx*3 + d, static_cast<unsigned long long>((long long) (-grad_delta*FIXED_EXPONENT)));
        }

    }
    atomicAdd(energy, kb/2*db*db);

}


template<typename RealType>
void __global__ k_harmonic_bond_jvp(
    const int B,     // number of bonds
    const double *coords,  
    const double *coords_tangent,  
    const double *params,  // [p, 2]
    const int *bond_idxs,    // [b, 2]
    double *grad_coords_primals,
    double *grad_coords_tangents,
    double *grad_params_primals,
    double *grad_params_tangents) {

    const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= B) {
        return;
    }

    int src_idx = bond_idxs[b_idx*2+0];
    int dst_idx = bond_idxs[b_idx*2+1];

    Surreal<RealType> dx[3];
    Surreal<RealType> d2ij(0.0, 0.0); // initialize your summed variables!
    for(int d=0; d < 3; d++) {
        Surreal<RealType> delta;
        delta.real = coords[src_idx*3+d] - coords[dst_idx*3+d];
        delta.imag = coords_tangent[src_idx*3+d] - coords_tangent[dst_idx*3+d];
        dx[d] = delta;
        d2ij += delta*delta;
    }

    int kb_idx = b_idx*2+0;
    int b0_idx = b_idx*2+1;

    RealType kb = params[kb_idx];
    RealType b0 = params[b0_idx];

    Surreal<RealType> dij = sqrt(d2ij);
    Surreal<RealType> db = dij - b0;

    for(int d=0; d < 3; d++) {
        Surreal<RealType> grad_delta = kb*db*dx[d]/dij;
        atomicAdd(grad_coords_primals + src_idx*3 + d, grad_delta.real);
        atomicAdd(grad_coords_primals + dst_idx*3 + d, -grad_delta.real);

        atomicAdd(grad_coords_tangents + src_idx*3 + d, grad_delta.imag);
        atomicAdd(grad_coords_tangents + dst_idx*3 + d, -grad_delta.imag);
    }

    // avoid writing out to the real parts if possible
    atomicAdd(grad_params_primals + kb_idx, (0.5*db*db).real);
    atomicAdd(grad_params_tangents + kb_idx, (0.5*db*db).imag);

    atomicAdd(grad_params_primals + b0_idx, (-kb*db).real);
    atomicAdd(grad_params_tangents + b0_idx, (-kb*db).imag);

}


template<typename RealType, int D>
void __global__ k_harmonic_angle_inference(
    const int A,     // number of bonds
    const double *coords,  // [n, 3]
    const double *params,  // [p, 2]
    const int *angle_idxs,    // [b, 3]
    unsigned long long *grad_coords,
    double *out_energy
) {

    const auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx >= A) {
        return;
    }

    int i_idx = angle_idxs[a_idx*3+0];
    int j_idx = angle_idxs[a_idx*3+1];
    int k_idx = angle_idxs[a_idx*3+2];

    RealType rij[MAXDIM];
    RealType rjk[MAXDIM];
    RealType nij = 0; // initialize your summed variables!
    RealType njk = 0; // initialize your summed variables!
    RealType top = 0;
    // this is a little confusing
    for(int d=0; d < MAXDIM; d++) {
        RealType vij = coords[j_idx*D+d] - coords[i_idx*D+d];
        RealType vjk = coords[j_idx*D+d] - coords[k_idx*D+d];

        rij[d] = vij;
        rjk[d] = vjk;
        nij += vij*vij;
        njk += vjk*vjk;

        top += vij*vjk;
    }

    nij = sqrt(nij);
    njk = sqrt(njk);

    RealType nijk = nij*njk;
    RealType n3ij = nij*nij*nij;
    RealType n3jk = njk*njk*njk;

    int ka_idx = a_idx*2+0;
    int a0_idx = a_idx*2+1;

    RealType ka = params[ka_idx];
    RealType a0 = params[a0_idx];

    RealType delta = top/nijk - cos(a0);

    for(int d=0; d < MAXDIM; d++) {
        RealType grad_i = ka*delta*(rij[d]*top/(n3ij*njk) + (-rjk[d])/nijk);
        atomicAdd(grad_coords + i_idx*D + d, static_cast<unsigned long long>((long long) (grad_i*FIXED_EXPONENT)));

        RealType grad_j = ka*delta*((-rij[d]*top/(n3ij*njk) + (-rjk[d])*top/(nij*n3jk) + (rij[d] + rjk[d])/nijk));
        atomicAdd(grad_coords + j_idx*D + d, static_cast<unsigned long long>((long long) (grad_j*FIXED_EXPONENT)));

        RealType grad_k = ka*delta*(-rij[d]/nijk + rjk[d]*top/(nij*n3jk));
        atomicAdd(grad_coords + k_idx*D + d, static_cast<unsigned long long>((long long) (grad_k*FIXED_EXPONENT)));
    }

    atomicAdd(out_energy, ka/2*delta*delta);

}


template<typename RealType, int D>
void __global__ k_harmonic_angle_jvp(
    const int A,     // number of bonds
    const double *coords,  // [n, 3]
    const double *coords_tangent,  // [n, 3]
    const double *params,  // [p, 2]
    const int *angle_idxs,    // [b, 3]
    double *grad_coords_primals,
    double *grad_coords_tangents,
    double *grad_params_primals,
    double *grad_params_tangents
) {

    const auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx >= A) {
        return;
    }

    int i_idx = angle_idxs[a_idx*3+0];
    int j_idx = angle_idxs[a_idx*3+1];
    int k_idx = angle_idxs[a_idx*3+2];

    Surreal<RealType> rij[MAXDIM];
    Surreal<RealType> rjk[MAXDIM];
    Surreal<RealType> nij(0.0, 0.0); // initialize your summed variables!
    Surreal<RealType> njk(0.0, 0.0); // initialize your summed variables!
    Surreal<RealType> top(0.0, 0.0);

    for(int d=0; d < MAXDIM; d++) {

        Surreal<RealType> vij;
        vij.real = coords[j_idx*D+d] - coords[i_idx*D+d];
        vij.imag = coords_tangent[j_idx*D+d] - coords_tangent[i_idx*D+d];

        Surreal<RealType> vjk;
        vjk.real = coords[j_idx*D+d] - coords[k_idx*D+d];
        vjk.imag = coords_tangent[j_idx*D+d] - coords_tangent[k_idx*D+d];

        rij[d] = vij;
        rjk[d] = vjk;

        nij += vij*vij;
        njk += vjk*vjk;

        top += vij*vjk;
    }

    nij = sqrt(nij);
    njk = sqrt(njk);

    Surreal<RealType> nijk = nij*njk;
    Surreal<RealType> n3ij = nij*nij*nij;
    Surreal<RealType> n3jk = njk*njk*njk;

    int ka_idx = a_idx*2+0;
    int a0_idx = a_idx*2+1;

    RealType ka = params[ka_idx];
    RealType a0 = params[a0_idx];

    Surreal<RealType> delta = top/nijk - cos(a0);

    for(int d=0; d < MAXDIM; d++) {
        Surreal<RealType> grad_i = ka*delta*(rij[d]*top/(n3ij*njk) + (-rjk[d])/nijk);
        atomicAdd(grad_coords_primals + i_idx*D + d, grad_i.real);
        atomicAdd(grad_coords_tangents + i_idx*D + d, grad_i.imag);

        Surreal<RealType> grad_j = ka*delta*((-rij[d]*top/(n3ij*njk) + (-rjk[d])*top/(nij*n3jk) + (rij[d] + rjk[d])/nijk));
        atomicAdd(grad_coords_primals + j_idx*D + d, grad_j.real);
        atomicAdd(grad_coords_tangents + j_idx*D + d, grad_j.imag);

        Surreal<RealType> grad_k = ka*delta*(-rij[d]/nijk + rjk[d]*top/(nij*n3jk));
        atomicAdd(grad_coords_primals + k_idx*D + d, grad_k.real);
        atomicAdd(grad_coords_tangents + k_idx*D + d, grad_k.imag);
    }

    // now we do dU/dp by hand
    Surreal<RealType> dka_grad = delta*delta/2;

    atomicAdd(grad_params_primals + ka_idx, dka_grad.real);
    atomicAdd(grad_params_tangents + ka_idx, dka_grad.imag);

    Surreal<RealType> da0_grad = delta*ka*sin(a0);

    atomicAdd(grad_params_primals + a0_idx, da0_grad.real);
    atomicAdd(grad_params_tangents + a0_idx, da0_grad.imag);
}


template<typename RealType>
inline __device__ RealType dot_product(
    const RealType a[3],
    const RealType b[3]) {

    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}


template<typename RealType>
inline __device__ void cross_product(
    const RealType a[3],
    const RealType b[3],
    RealType c[3]) {
    // fix this one indexed garbage later
    c[1-1] = a[2-1]*b[3-1] - a[3-1]*b[2-1];
    c[2-1] = a[3-1]*b[1-1] - a[1-1]*b[3-1];
    c[3-1] = a[1-1]*b[2-1] - a[2-1]*b[1-1];

}


template<typename RealType, int D>
void __global__ k_periodic_torsion_inference(
    const int T,     // number of bonds
    const double *coords,  // [n, 3]
    const double *params,  // [p, 3]
    const int *torsion_idxs,    // [b, 4]
    unsigned long long *grad_coords,
    double *energy
) {

    const auto t_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(t_idx >= T) {
        return;
    }

    int i_idx = torsion_idxs[t_idx*4+0];
    int j_idx = torsion_idxs[t_idx*4+1];
    int k_idx = torsion_idxs[t_idx*4+2];
    int l_idx = torsion_idxs[t_idx*4+3];

    RealType rij[3];
    RealType rkj[3];
    RealType rkl[3];

    RealType rkj_norm_square = 0;

    // (todo) cap to three dims, while keeping stride at 4
    for(int d=0; d < 3; d++) {
        RealType vij = coords[j_idx*D+d] - coords[i_idx*D+d];
        RealType vkj = coords[j_idx*D+d] - coords[k_idx*D+d];
        RealType vkl = coords[l_idx*D+d] - coords[k_idx*D+d];
        rij[d] = vij;
        rkj[d] = vkj;
        rkl[d] = vkl;
        rkj_norm_square += vkj*vkj;
    }

    RealType rkj_norm = sqrt(rkj_norm_square);
    RealType n1[3], n2[3];

    cross_product(rij, rkj, n1);
    cross_product(rkj, rkl, n2);

    RealType n1_norm_square, n2_norm_square;

    n1_norm_square = dot_product(n1, n1);
    n2_norm_square = dot_product(n2, n2);

    RealType n3[3];
    cross_product(n1, n2, n3);

    RealType d_angle_dR0[3];
    RealType d_angle_dR3[3];
    RealType d_angle_dR1[3];
    RealType d_angle_dR2[3];

    RealType rij_dot_rkj = dot_product(rij, rkj);
    RealType rkl_dot_rkj = dot_product(rkl, rkj);

    for(int d=0; d < 3; d++) {
        d_angle_dR0[d] = rkj_norm/n1_norm_square * n1[d];
        d_angle_dR3[d] = -rkj_norm/n2_norm_square * n2[d];
        d_angle_dR1[d] = (rij_dot_rkj/rkj_norm_square - 1)*d_angle_dR0[d] - d_angle_dR3[d]*rkl_dot_rkj/rkj_norm_square;
        d_angle_dR2[d] = (rkl_dot_rkj/rkj_norm_square - 1)*d_angle_dR3[d] - d_angle_dR0[d]*rij_dot_rkj/rkj_norm_square;
    }

    RealType rkj_n = sqrt(dot_product(rkj, rkj));

    for(int d=0; d < 3; d++) {
        rkj[d] /= rkj_n;
    }

    RealType y = dot_product(n3, rkj);
    RealType x = dot_product(n1, n2);
    RealType angle = atan2(y, x);

    int kt_idx = t_idx*3+0;
    int phase_idx = t_idx*3+1;
    int period_idx = t_idx*3+2;

    RealType kt = params[kt_idx];
    RealType phase = params[phase_idx];
    RealType period = params[period_idx];

    RealType prefactor = kt*sin(period*angle - phase)*period;

    for(int d=0; d < 3; d++) {
        atomicAdd(grad_coords + i_idx*D + d, static_cast<unsigned long long>((long long) (d_angle_dR0[d] * prefactor * FIXED_EXPONENT)));
        atomicAdd(grad_coords + j_idx*D + d, static_cast<unsigned long long>((long long) (d_angle_dR1[d] * prefactor * FIXED_EXPONENT)));
        atomicAdd(grad_coords + k_idx*D + d, static_cast<unsigned long long>((long long) (d_angle_dR2[d] * prefactor * FIXED_EXPONENT)));
        atomicAdd(grad_coords + l_idx*D + d, static_cast<unsigned long long>((long long) (d_angle_dR3[d] * prefactor * FIXED_EXPONENT)));
    }

    atomicAdd(energy, kt*(1+cos(period*angle - phase)));

}


template<typename RealType, int D>
void __global__ k_periodic_torsion_jvp(
    const int T,     // number of bonds
    const double *coords,  // [n, 3]
    const double *coords_tangent, 
    const double *params,  // [p, 3]
    const int *torsion_idxs,    // [b, 4]
    double *grad_coords_primals,
    double *grad_coords_tangents,
    double *grad_params_primals,
    double *grad_params_tangents
) {

    const auto t_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(t_idx >= T) {
        return;
    }

    int i_idx = torsion_idxs[t_idx*4+0];
    int j_idx = torsion_idxs[t_idx*4+1];
    int k_idx = torsion_idxs[t_idx*4+2];
    int l_idx = torsion_idxs[t_idx*4+3];

    Surreal<RealType> rij[3];
    Surreal<RealType> rkj[3];
    Surreal<RealType> rkl[3];

    Surreal<RealType> rkj_norm_square(0.0, 0.0);

    // (todo) cap to three dims, while keeping stride at 4
    for(int d=0; d < 3; d++) {
        Surreal<RealType> vij; vij.real = coords[j_idx*D+d] - coords[i_idx*D+d]; vij.imag = coords_tangent[j_idx*D+d] - coords_tangent[i_idx*D+d];
        Surreal<RealType> vkj; vkj.real = coords[j_idx*D+d] - coords[k_idx*D+d]; vkj.imag = coords_tangent[j_idx*D+d] - coords_tangent[k_idx*D+d];
        Surreal<RealType> vkl; vkl.real = coords[l_idx*D+d] - coords[k_idx*D+d]; vkl.imag = coords_tangent[l_idx*D+d] - coords_tangent[k_idx*D+d];
        rij[d] = vij;
        rkj[d] = vkj;
        rkl[d] = vkl;
        rkj_norm_square += vkj*vkj;
    }

    Surreal<RealType> rkj_norm = sqrt(rkj_norm_square);
    Surreal<RealType> n1[3], n2[3];

    cross_product(rij, rkj, n1);
    cross_product(rkj, rkl, n2);

    Surreal<RealType> n1_norm_square, n2_norm_square;

    n1_norm_square = dot_product(n1, n1);
    n2_norm_square = dot_product(n2, n2);

    Surreal<RealType> n3[3];
    cross_product(n1, n2, n3);

    Surreal<RealType> d_angle_dR0[3];
    Surreal<RealType> d_angle_dR3[3];
    Surreal<RealType> d_angle_dR1[3];
    Surreal<RealType> d_angle_dR2[3];

    Surreal<RealType> rij_dot_rkj = dot_product(rij, rkj);
    Surreal<RealType> rkl_dot_rkj = dot_product(rkl, rkj);

    for(int d=0; d < 3; d++) {
        d_angle_dR0[d] = rkj_norm/n1_norm_square * n1[d];
        d_angle_dR3[d] = -rkj_norm/n2_norm_square * n2[d];
        d_angle_dR1[d] = (rij_dot_rkj/rkj_norm_square - 1)*d_angle_dR0[d] - d_angle_dR3[d]*rkl_dot_rkj/rkj_norm_square;
        d_angle_dR2[d] = (rkl_dot_rkj/rkj_norm_square - 1)*d_angle_dR3[d] - d_angle_dR0[d]*rij_dot_rkj/rkj_norm_square;
    }

    Surreal<RealType> rkj_n = sqrt(dot_product(rkj, rkj));

    for(int d=0; d < 3; d++) {
        rkj[d] /= rkj_n;
    }

    Surreal<RealType> y = dot_product(n3, rkj);
    Surreal<RealType> x = dot_product(n1, n2);
    Surreal<RealType> angle = atan2(y, x);

    int kt_idx = t_idx*3+0;
    int phase_idx = t_idx*3+1;
    int period_idx = t_idx*3+2;

    RealType kt = params[kt_idx];
    RealType phase = params[phase_idx];
    RealType period = params[period_idx];

    Surreal<RealType> prefactor = kt*sin(period*angle - phase)*period;

    for(int d=0; d < 3; d++) {
        atomicAdd(grad_coords_primals + i_idx*D + d, (d_angle_dR0[d] * prefactor).real);
        atomicAdd(grad_coords_tangents + i_idx*D + d, (d_angle_dR0[d] * prefactor).imag);
        atomicAdd(grad_coords_primals + j_idx*D + d, (d_angle_dR1[d] * prefactor).real);
        atomicAdd(grad_coords_tangents + j_idx*D + d, (d_angle_dR1[d] * prefactor).imag);
        atomicAdd(grad_coords_primals + k_idx*D + d, (d_angle_dR2[d] * prefactor).real);
        atomicAdd(grad_coords_tangents + k_idx*D + d, (d_angle_dR2[d] * prefactor).imag);
        atomicAdd(grad_coords_primals + l_idx*D + d, (d_angle_dR3[d] * prefactor).real);
        atomicAdd(grad_coords_tangents + l_idx*D + d, (d_angle_dR3[d] * prefactor).imag);
    }

    Surreal<RealType> du_dkt = 1 + cos(period*angle - phase);
    Surreal<RealType> du_dphase = kt*sin(period*angle - phase);
    Surreal<RealType> du_dperiod = -kt*sin(period*angle - phase)*angle;

    atomicAdd(grad_params_primals + kt_idx, du_dkt.real);
    atomicAdd(grad_params_tangents + kt_idx, du_dkt.imag);
    atomicAdd(grad_params_primals + phase_idx, du_dphase.real);
    atomicAdd(grad_params_tangents + phase_idx, du_dphase.imag);
    atomicAdd(grad_params_primals + period_idx, du_dperiod.real);
    atomicAdd(grad_params_tangents + period_idx, du_dperiod.imag);

}