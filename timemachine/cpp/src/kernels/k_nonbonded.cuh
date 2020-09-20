#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#define WARPSIZE 32

#define PI 3.141592653589793115997963468544185161

template<typename RealType>
void __global__ k_reduce_buffer(
    int N,
    RealType *d_buffer,
    RealType *d_sum) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    RealType elem = idx < N ? d_buffer[idx] : 0;

    atomicAdd(d_sum, elem);

};

template <typename RealType>
void __global__ k_nonbonded(
    const int N,
    const double * __restrict__ coords,
    const double * __restrict__ params, // [N]
    const double * __restrict__ box,
    const double lambda,
    const int * __restrict__ lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    const int * __restrict__ ixn_tiles,
    const unsigned int * __restrict__ ixn_atoms,
    unsigned long long * __restrict__ du_dx,
    double * __restrict__ du_dp,
    double * __restrict__ du_dl_buffer,
    double * __restrict__ u_buffer) {

    int tile_idx = blockIdx.x;

    RealType box_x = box[0*3+0];
    RealType box_y = box[1*3+1];
    RealType box_z = box[2*3+2];

    RealType inv_box_x = 1/box_x;
    RealType inv_box_y = 1/box_y;
    RealType inv_box_z = 1/box_z;

    int row_block_idx = ixn_tiles[tile_idx];

    int atom_i_idx = row_block_idx*32 + threadIdx.x;
    int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;

    RealType ci_x = atom_i_idx < N ? coords[atom_i_idx*3+0] : 0;
    RealType ci_y = atom_i_idx < N ? coords[atom_i_idx*3+1] : 0;
    RealType ci_z = atom_i_idx < N ? coords[atom_i_idx*3+2] : 0;
    RealType gi_x = 0;
    RealType gi_y = 0;
    RealType gi_z = 0;
    RealType du_dl_i = 0;

    int charge_param_idx_i = atom_i_idx*3 + 0;
    int lj_param_idx_sig_i = atom_i_idx*3 + 1;
    int lj_param_idx_eps_i = atom_i_idx*3 + 2;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 0;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    RealType g_qi = 0;
    RealType g_sigi = 0;
    RealType g_epsi = 0;

    int atom_j_idx = ixn_atoms[tile_idx*32 + threadIdx.x];
    int lambda_offset_j = atom_j_idx < N ? lambda_offset_idxs[atom_j_idx] : 0;

    RealType cj_x = atom_j_idx < N ? coords[atom_j_idx*3+0] : 0;
    RealType cj_y = atom_j_idx < N ? coords[atom_j_idx*3+1] : 0;
    RealType cj_z = atom_j_idx < N ? coords[atom_j_idx*3+2] : 0;
    RealType gj_x = 0;
    RealType gj_y = 0;
    RealType gj_z = 0;
    RealType du_dl_j = 0;

    int charge_param_idx_j = atom_j_idx*3 + 0;
    int lj_param_idx_sig_j = atom_j_idx*3 + 1;
    int lj_param_idx_eps_j = atom_j_idx*3 + 2;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 0;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

    RealType g_qj = 0;
    RealType g_sigj = 0;
    RealType g_epsj = 0;

    RealType cutoff_squared = cutoff*cutoff;

    RealType energy = 0;

    RealType real_lambda = static_cast<RealType>(lambda);
    RealType real_beta = static_cast<RealType>(beta);

    const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
    // #pragma unroll
    for(int round = 0; round < 32; round++) {

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;

        delta_x -= box_x*nearbyint(delta_x*inv_box_x);
        delta_y -= box_y*nearbyint(delta_y*inv_box_y);
        delta_z -= box_z*nearbyint(delta_z*inv_box_z);

        RealType delta_w = (lambda_offset_i - lambda_offset_j)*real_lambda;

        RealType d2ij = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + delta_w*delta_w;

        // remember the eps_i != 0
        if(d2ij < cutoff_squared  && atom_j_idx > atom_i_idx && atom_j_idx < N && atom_i_idx < N) {

            // electrostatics
            RealType dij = sqrt(d2ij);
            RealType inv_dij = 1/dij;

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType ebd = erfc(real_beta*dij);
            RealType qij = qi*qj;

            RealType es_prefactor = qij*(-2*real_beta*exp(-real_beta*real_beta*dij*dij)/(sqrt(static_cast<RealType>(PI))*dij) - ebd*inv_d2ij)*inv_dij;

            // lennard jones
            RealType inv_d3ij = inv_dij*inv_d2ij;
            RealType inv_d4ij = inv_d2ij*inv_d2ij;
            RealType inv_d6ij = inv_d4ij*inv_d2ij;
            RealType inv_d8ij = inv_d4ij*inv_d4ij;

            // lennard jones force
            RealType eps_ij = sqrt(eps_i * eps_j);
            RealType sig_ij = (sig_i + sig_j)/2;

            RealType sig2 = sig_ij*sig_ij;
            RealType sig4 = sig2*sig2;
            RealType sig5 = sig4*sig_ij;
            RealType sig6 = sig4*sig2;

            RealType sig6_inv_d6ij = sig6*inv_d6ij;
            RealType sig6_inv_d8ij = sig6*inv_d8ij;

            RealType lj_prefactor = 24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1);

            // accumulate
            gi_x += (es_prefactor-lj_prefactor)*delta_x;
            gi_y += (es_prefactor-lj_prefactor)*delta_y;
            gi_z += (es_prefactor-lj_prefactor)*delta_z;

            gj_x -= (es_prefactor-lj_prefactor)*delta_x;
            gj_y -= (es_prefactor-lj_prefactor)*delta_y;
            gj_z -= (es_prefactor-lj_prefactor)*delta_z;

            du_dl_i += (es_prefactor-lj_prefactor)*delta_w*lambda_offset_i;
            du_dl_j -= (es_prefactor-lj_prefactor)*delta_w*lambda_offset_j;

            energy += qij*inv_dij*ebd + 4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;

            g_qi += qj*inv_dij*ebd;
            g_qj += qi*inv_dij*ebd;

            // the derivative is undefined if epsilons are zero.
            if(eps_i != 0 && eps_j != 0) {

                RealType eps_grad = 2*sig6_inv_d6ij*(sig6_inv_d6ij-1)/eps_ij;
                g_epsi += eps_grad*eps_j;
                g_epsj += eps_grad*eps_i;

                RealType sig_grad = 12*eps_ij*sig5*inv_d6ij*(2*sig6_inv_d6ij-1);
                g_sigi += sig_grad;
                g_sigj += sig_grad;

            }

        }

        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane); // we can pre-compute this probably
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        cj_x = __shfl_sync(0xffffffff, cj_x, srcLane); // needs to support real
        cj_y = __shfl_sync(0xffffffff, cj_y, srcLane); // needs to support real
        cj_z = __shfl_sync(0xffffffff, cj_z, srcLane); // needs to support real
        gj_x = __shfl_sync(0xffffffff, gj_x, srcLane);
        gj_y = __shfl_sync(0xffffffff, gj_y, srcLane);
        gj_z = __shfl_sync(0xffffffff, gj_z, srcLane);
        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane); // this also can be optimized away
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    // these reduction buffers are really tricky
    if(du_dx) {
        if(atom_i_idx < N) {
            atomicAdd(du_dx + atom_i_idx*3 + 0, static_cast<unsigned long long>((long long) (gi_x*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_i_idx*3 + 1, static_cast<unsigned long long>((long long) (gi_y*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_i_idx*3 + 2, static_cast<unsigned long long>((long long) (gi_z*FIXED_EXPONENT)));
        }
        if(atom_j_idx < N) {
            atomicAdd(du_dx + atom_j_idx*3 + 0, static_cast<unsigned long long>((long long) (gj_x*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_j_idx*3 + 1, static_cast<unsigned long long>((long long) (gj_y*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_j_idx*3 + 2, static_cast<unsigned long long>((long long) (gj_z*FIXED_EXPONENT)));
        }
    }

    if(du_dp) {

        if(atom_i_idx < N) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
            atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
            atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);
        }

        if(atom_j_idx < N) {
            atomicAdd(du_dp + charge_param_idx_j, g_qj);
            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
        }

    }

    // these are buffered and then reduced to avoid massive conflicts
    if(du_dl_buffer) {
        if(atom_i_idx < N) {
            atomicAdd(du_dl_buffer + atom_i_idx, du_dl_i + du_dl_j);
        }
    }

    if(u_buffer) {
        if(atom_i_idx < N) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }
    }

}


template<typename RealType>
void __global__ k_nonbonded_exclusions(
    const int E, // number of exclusions
    const double *coords,
    const double *params,
    const double *box,
    const double lambda,
    const int *lambda_offset_idxs, // 0 or 1, if we alolw this atom to be decoupled
    const int *exclusion_idxs, // [E, 2] pair-list of atoms to be excluded
    const double *scales, // [E]
    const double beta,
    const double cutoff,
    unsigned long long *du_dx,
    double *du_dp,
    double *du_dl_buffer,
    double *u_buffer) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];

    RealType ci_x = coords[atom_i_idx*3+0];
    RealType ci_y = coords[atom_i_idx*3+1];
    RealType ci_z = coords[atom_i_idx*3+2];
    RealType gi_x = 0;
    RealType gi_y = 0;
    RealType gi_z = 0;
    RealType du_dl_i = 0;

    int charge_param_idx_i = atom_i_idx*3 + 0;
    int lj_param_idx_sig_i = atom_i_idx*3 + 1;
    int lj_param_idx_eps_i = atom_i_idx*3 + 2;

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];

    RealType g_qi = 0;
    RealType g_sigi = 0;
    RealType g_epsi = 0;

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];

    RealType cj_x = coords[atom_j_idx*3+0];
    RealType cj_y = coords[atom_j_idx*3+1];
    RealType cj_z = coords[atom_j_idx*3+2];
    RealType gj_x = 0;
    RealType gj_y = 0;
    RealType gj_z = 0;
    RealType du_dl_j = 0;

    int charge_param_idx_j = atom_j_idx*3+0;
    int lj_param_idx_sig_j = atom_j_idx*3 + 1;
    int lj_param_idx_eps_j = atom_j_idx*3 + 2;

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];

    RealType g_qj = 0;
    RealType g_sigj = 0;
    RealType g_epsj = 0;

    RealType real_lambda = static_cast<RealType>(lambda);
    RealType real_beta = static_cast<RealType>(beta);
    RealType cutoff_squared = cutoff*cutoff;

    RealType charge_scale = scales[e_idx*2 + 0];
    RealType lj_scale = scales[e_idx*2 + 1];

    RealType box_x = box[0*3+0];
    RealType box_y = box[1*3+1];
    RealType box_z = box[2*3+2];

    RealType inv_box_x = 1/box_x;
    RealType inv_box_y = 1/box_y;
    RealType inv_box_z = 1/box_z;

    RealType delta_x = ci_x - cj_x;
    RealType delta_y = ci_y - cj_y;
    RealType delta_z = ci_z - cj_z;

    delta_x -= box_x*nearbyint(delta_x*inv_box_x);
    delta_y -= box_y*nearbyint(delta_y*inv_box_y);
    delta_z -= box_z*nearbyint(delta_z*inv_box_z);

    RealType delta_w = (lambda_offset_i - lambda_offset_j)*real_lambda;

    RealType d2ij = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + delta_w*delta_w;

    RealType energy = 0;

    if(d2ij < cutoff_squared) {

        RealType dij = sqrt(d2ij);
        RealType inv_dij = 1/dij;

        RealType inv_d2ij = inv_dij*inv_dij;
        RealType ebd = erfc(real_beta*dij);
        RealType qij = qi*qj;

        RealType es_prefactor = charge_scale*qij*(-2*real_beta*exp(-real_beta*real_beta*dij*dij)/(sqrt(static_cast<RealType>(PI))*dij) - ebd*inv_d2ij)*inv_dij;

        // lennard jones
        RealType inv_d3ij = inv_dij*inv_d2ij;
        RealType inv_d4ij = inv_d2ij*inv_d2ij;
        RealType inv_d6ij = inv_d4ij*inv_d2ij;
        RealType inv_d8ij = inv_d4ij*inv_d4ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        RealType sig2 = sig_ij*sig_ij;
        RealType sig4 = sig2*sig2;
        RealType sig5 = sig4*sig_ij;
        RealType sig6 = sig4*sig2;

        RealType sig6_inv_d6ij = sig6*inv_d6ij;
        RealType sig6_inv_d8ij = sig6*inv_d8ij;

        RealType lj_prefactor = lj_scale*24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1);

        gi_x -= (es_prefactor-lj_prefactor)*delta_x;
        gi_y -= (es_prefactor-lj_prefactor)*delta_y;
        gi_z -= (es_prefactor-lj_prefactor)*delta_z;

        gj_x += (es_prefactor-lj_prefactor)*delta_x;
        gj_y += (es_prefactor-lj_prefactor)*delta_y;
        gj_z += (es_prefactor-lj_prefactor)*delta_z;

        du_dl_i -= (es_prefactor-lj_prefactor)*delta_w*lambda_offset_i;
        du_dl_j += (es_prefactor-lj_prefactor)*delta_w*lambda_offset_j;

        energy -= charge_scale*qij*inv_dij*ebd + lj_scale*4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;

        g_qi -= charge_scale*qj*inv_dij*ebd;
        g_qj -= charge_scale*qi*inv_dij*ebd;

        if(eps_i != 0 && eps_j != 0) {

            RealType eps_grad = lj_scale*2*sig6_inv_d6ij*(sig6_inv_d6ij-1)/eps_ij;
            g_epsi -= eps_grad*eps_j;
            g_epsj -= eps_grad*eps_i;

            RealType sig_grad = lj_scale*12*eps_ij*sig5*inv_d6ij*(2*sig6_inv_d6ij-1);
            g_sigi -= sig_grad;
            g_sigj -= sig_grad;

        }

        // these reduction buffers are really tricky
        if(du_dx) {
            atomicAdd(du_dx + atom_i_idx*3 + 0, static_cast<unsigned long long>((long long) (gi_x*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_i_idx*3 + 1, static_cast<unsigned long long>((long long) (gi_y*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_i_idx*3 + 2, static_cast<unsigned long long>((long long) (gi_z*FIXED_EXPONENT)));

            atomicAdd(du_dx + atom_j_idx*3 + 0, static_cast<unsigned long long>((long long) (gj_x*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_j_idx*3 + 1, static_cast<unsigned long long>((long long) (gj_y*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_j_idx*3 + 2, static_cast<unsigned long long>((long long) (gj_z*FIXED_EXPONENT)));
        }

        if(du_dp) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
            atomicAdd(du_dp + charge_param_idx_j, g_qj);

            atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
            atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);

            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
        }

        if(du_dl_buffer) {
            atomicAdd(du_dl_buffer + atom_i_idx, du_dl_i + du_dl_j);
        }

        if(u_buffer) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }

    }

}