#include "../fixed_point.hpp"

template<typename RealType, int D>
void __global__ k_harmonic_angle_inference(
    const int A,     // number of bonds
    const double *coords,  // [N, 3]
    const double *params,  // [P, 2]
    const int *angle_idxs,    // [A, 3]
    unsigned long long *du_dx,
    double *du_dp,
    double *u) {

    const auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx >= A) {
        return;
    }

    int i_idx = angle_idxs[a_idx*3+0];
    int j_idx = angle_idxs[a_idx*3+1];
    int k_idx = angle_idxs[a_idx*3+2];

    RealType rij[3];
    RealType rjk[3];
    RealType nij = 0; // initialize your summed variables!
    RealType njk = 0; // initialize your summed variables!
    RealType top = 0;
    // this is a little confusing
    for(int d=0; d < 3; d++) {
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

    if(du_dx) {
        for(int d=0; d < 3; d++) {
            RealType grad_i = ka*delta*(rij[d]*top/(n3ij*njk) + (-rjk[d])/nijk);
            atomicAdd(du_dx + i_idx*D + d, static_cast<unsigned long long>((long long) (grad_i*FIXED_EXPONENT)));

            RealType grad_j = ka*delta*((-rij[d]*top/(n3ij*njk) + (-rjk[d])*top/(nij*n3jk) + (rij[d] + rjk[d])/nijk));
            atomicAdd(du_dx + j_idx*D + d, static_cast<unsigned long long>((long long) (grad_j*FIXED_EXPONENT)));

            RealType grad_k = ka*delta*(-rij[d]/nijk + rjk[d]*top/(nij*n3jk));
            atomicAdd(du_dx + k_idx*D + d, static_cast<unsigned long long>((long long) (grad_k*FIXED_EXPONENT)));
        }
    }

    if(du_dp) {
        RealType dka_grad = delta*delta/2;
        atomicAdd(du_dp + ka_idx, dka_grad);
        RealType da0_grad = delta*ka*sin(a0);
        atomicAdd(du_dp + a0_idx, da0_grad);        
    }

    if(u) {
        atomicAdd(u, ka/2*delta*delta);        
    }

}


// template<typename RealType, int D>
// void __global__ k_harmonic_angle_jvp(
//     const int A,     // number of bonds
//     const double *coords,  // [n, 3]
//     const double *coords_tangent,  // [n, 3]
//     const double *params,  // [p, 2]
//     const int *angle_idxs,    // [b, 3]
//     double *grad_coords_primals,
//     double *grad_coords_tangents,
//     double *grad_params_primals,
//     double *grad_params_tangents
// ) {

//     const auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

//     if(a_idx >= A) {
//         return;
//     }

//     int i_idx = angle_idxs[a_idx*3+0];
//     int j_idx = angle_idxs[a_idx*3+1];
//     int k_idx = angle_idxs[a_idx*3+2];

//     Surreal<RealType> rij[MAXDIM];
//     Surreal<RealType> rjk[MAXDIM];
//     Surreal<RealType> nij(0.0, 0.0); // initialize your summed variables!
//     Surreal<RealType> njk(0.0, 0.0); // initialize your summed variables!
//     Surreal<RealType> top(0.0, 0.0);

//     for(int d=0; d < MAXDIM; d++) {

//         Surreal<RealType> vij;
//         vij.real = coords[j_idx*D+d] - coords[i_idx*D+d];
//         vij.imag = coords_tangent[j_idx*D+d] - coords_tangent[i_idx*D+d];

//         Surreal<RealType> vjk;
//         vjk.real = coords[j_idx*D+d] - coords[k_idx*D+d];
//         vjk.imag = coords_tangent[j_idx*D+d] - coords_tangent[k_idx*D+d];

//         rij[d] = vij;
//         rjk[d] = vjk;

//         nij += vij*vij;
//         njk += vjk*vjk;

//         top += vij*vjk;
//     }

//     nij = sqrt(nij);
//     njk = sqrt(njk);

//     Surreal<RealType> nijk = nij*njk;
//     Surreal<RealType> n3ij = nij*nij*nij;
//     Surreal<RealType> n3jk = njk*njk*njk;

//     int ka_idx = a_idx*2+0;
//     int a0_idx = a_idx*2+1;

//     RealType ka = params[ka_idx];
//     RealType a0 = params[a0_idx];

//     Surreal<RealType> delta = top/nijk - cos(a0);

//     for(int d=0; d < MAXDIM; d++) {
//         Surreal<RealType> grad_i = ka*delta*(rij[d]*top/(n3ij*njk) + (-rjk[d])/nijk);
//         atomicAdd(grad_coords_primals + i_idx*D + d, grad_i.real);
//         atomicAdd(grad_coords_tangents + i_idx*D + d, grad_i.imag);

//         Surreal<RealType> grad_j = ka*delta*((-rij[d]*top/(n3ij*njk) + (-rjk[d])*top/(nij*n3jk) + (rij[d] + rjk[d])/nijk));
//         atomicAdd(grad_coords_primals + j_idx*D + d, grad_j.real);
//         atomicAdd(grad_coords_tangents + j_idx*D + d, grad_j.imag);

//         Surreal<RealType> grad_k = ka*delta*(-rij[d]/nijk + rjk[d]*top/(nij*n3jk));
//         atomicAdd(grad_coords_primals + k_idx*D + d, grad_k.real);
//         atomicAdd(grad_coords_tangents + k_idx*D + d, grad_k.imag);
//     }

//     // now we do dU/dp by hand
//     Surreal<RealType> dka_grad = delta*delta/2;

//     atomicAdd(grad_params_primals + ka_idx, dka_grad.real);
//     atomicAdd(grad_params_tangents + ka_idx, dka_grad.imag);

//     Surreal<RealType> da0_grad = delta*ka*sin(a0);

//     atomicAdd(grad_params_primals + a0_idx, da0_grad.real);
//     atomicAdd(grad_params_tangents + a0_idx, da0_grad.imag);
// }