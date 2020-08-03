#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#include "k_periodic_utils.cuh"

template <typename RealType>
void __global__ k_electrostatics_jvp(
    const int N,
    const double *coords, // maybe Surreal or Real
    const double *coords_tangent, // maybe Surreal or Real
    const double lambda_primal,
    const double lambda_tangent,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double *charge_params,
    // const double *lj_params, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    double *grad_coords_primals,
    double *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    double *grad_charge_params_primals,
    double *grad_charge_params_tangents) {
    // double *grad_lj_params_primals,
    // double *grad_lj_params_tangents) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

    RealType block_d2ij = 0; 
    for(int d=0; d < 3; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];
        RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;

    int lambda_plane_i = 0;
    int lambda_offset_i = 0;

    if(atom_i_idx < N) {
        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    }

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3];

    for(int d=0; d < 3; d++) {
        gi[d].real = 0.0;
        gi[d].imag = 0.0;
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangent[atom_i_idx*3+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? atom_i_idx : 0;

    RealType qi = atom_i_idx < N ? charge_params[charge_param_idx_i] : 0;

    Surreal<RealType> g_qi(0.0, 0.0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;

    if(atom_j_idx < N) {
        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    }

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3];

    for(int d=0; d < 3; d++) {
        gj[d].real = 0.0;
        gj[d].imag = 0.0;
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangent[atom_j_idx*3+d] : 0;
    }

    int charge_param_idx_j = atom_j_idx < N ? atom_j_idx : 0;

    RealType qj = atom_j_idx < N ? charge_params[charge_param_idx_j] : 0;

    Surreal<RealType> g_qj(0.0, 0.0);

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    for(int round = 0; round < 32; round++) {

        Surreal<RealType> dxs[4];
        Surreal<RealType> d2ij(0,0);
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
            d2ij += dxs[d]*dxs[d];
        }

        Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
        d2ij += delta_lambda * delta_lambda;

        if(atom_j_idx < atom_i_idx && d2ij.real < cutoff*cutoff && atom_j_idx < N && atom_i_idx < N) {

            Surreal<RealType> inv_dij = rsqrt(d2ij);
            Surreal<RealType> inv_d2ij = 1/d2ij;
            Surreal<RealType> inv_d3ij = inv_d2ij*inv_dij;
            Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

            for(int d=0; d < 3; d++) {
                gi[d] -= es_grad_prefactor * dxs[d];
                gj[d] += es_grad_prefactor * dxs[d];
            }

            // Charge
            g_qi += qj*inv_dij;
            g_qj += qi*inv_dij;

        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
    }

    // we should always accumulate in double precision

    // (ytz): we don't care about deterministic atomics that much when
    // doing reverse mode since we only ever have to do it once.
    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(grad_coords_primals + atom_i_idx*3 + d, gi[d].real);
            atomicAdd(grad_coords_tangents + atom_i_idx*3 + d, gi[d].imag);
        }
        if(atom_j_idx < N) {
            atomicAdd(grad_coords_primals + atom_j_idx*3 + d, gj[d].real);
            atomicAdd(grad_coords_tangents + atom_j_idx*3 + d, gj[d].imag);
        }
    }  

    if(atom_i_idx < N) {
        atomicAdd(grad_charge_params_primals + charge_param_idx_i, g_qi.real);
        atomicAdd(grad_charge_params_tangents + charge_param_idx_i, g_qi.imag);
    }

    if(atom_j_idx < N) {
        atomicAdd(grad_charge_params_primals + charge_param_idx_j, g_qj.real);
        atomicAdd(grad_charge_params_tangents + charge_param_idx_j, g_qj.imag);
    }
}


template<typename RealType>
void __global__ k_electrostatics_exclusion_jvp(
    const int E, // number of exclusions
    const double *coords,
    const double *coords_tangent,
    const double lambda_primal,
    const double lambda_tangent,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const double *charge_scales, // [E]
    const double *charge_params, // [N]
    const double cutoff,
    double *grad_coords_primals,
    double *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    double *grad_charge_params_primals,
    double *grad_charge_params_tangents) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    int lambda_plane_i = lambda_plane_idxs[atom_i_idx];
    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        gi[d].real = 0;
        gi[d].imag = 0;
        ci[d].real = coords[atom_i_idx*3+d];
        ci[d].imag = coords_tangent[atom_i_idx*3+d];
    }
    int charge_param_idx_i = atom_i_idx;

    RealType qi = charge_params[charge_param_idx_i];

    Surreal<RealType> g_qi(0.0, 0.0);

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    int lambda_plane_j = lambda_plane_idxs[atom_j_idx];
    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        gj[d].real = 0;
        gj[d].imag = 0;
        cj[d].real = coords[atom_j_idx*3+d];
        cj[d].imag = coords_tangent[atom_j_idx*3+d];
    }

    int charge_param_idx_j = atom_j_idx;

    RealType qj = charge_params[charge_param_idx_j];

    Surreal<RealType> g_qj(0.0, 0.0);
    RealType charge_scale = charge_scales[e_idx];
    
    Surreal<RealType> d2ij(0.0, 0.0);
    #pragma unroll
    for(int d=0; d < 3; d++) {
        Surreal<RealType> dx = ci[d] - cj[d];
        d2ij += dx*dx;
    }

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;

    d2ij += delta_lambda * delta_lambda;

    if(d2ij.real < cutoff*cutoff) {

        Surreal<RealType> inv_dij = rsqrt(d2ij);
        Surreal<RealType> inv_d2ij = 1/d2ij;
        Surreal<RealType> inv_d3ij = inv_dij*inv_d2ij;
        Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

        #pragma unroll
        for(int d=0; d < 3; d++) {
            Surreal<RealType> dx = ci[d] - cj[d];
            gi[d] += charge_scale * es_grad_prefactor * dx;
            gj[d] -= charge_scale * es_grad_prefactor * dx;
        }

        for(int d=0; d < 3; d++) {
            atomicAdd(grad_coords_primals + atom_i_idx*3 + d, gi[d].real);
            atomicAdd(grad_coords_tangents + atom_i_idx*3 + d, gi[d].imag);
            atomicAdd(grad_coords_primals + atom_j_idx*3 + d, gj[d].real);
            atomicAdd(grad_coords_tangents + atom_j_idx*3 + d, gj[d].imag);
        }  

        // dE_dp 
        // Charge
        g_qi += qj*inv_dij;
        g_qj += qi*inv_dij;

        atomicAdd(grad_charge_params_primals + charge_param_idx_i, -charge_scale*g_qi.real);
        atomicAdd(grad_charge_params_primals + charge_param_idx_j, -charge_scale*g_qj.real);

        atomicAdd(grad_charge_params_tangents + charge_param_idx_i, -charge_scale*g_qi.imag);
        atomicAdd(grad_charge_params_tangents + charge_param_idx_j, -charge_scale*g_qj.imag);


    }

}
