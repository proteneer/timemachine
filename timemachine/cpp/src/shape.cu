#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "shape.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"

#define PI 3.141592653589793115997963468544185161

namespace jankmachine {


// tbd chain rule for parameter derivatives
void __global__ k_reduce_dvol_dx_buffers(
    const int N,
    const double *f_buffer,
    const double *df_dx_buffer,
    const double *g_buffer,
    const double *dg_dx_buffer,
    const double *h_buffer,
    const double *dh_dx_buffer,
    const double k,
    unsigned long long *du_dx) {

    const int idx = blockDim.x*blockIdx.x + threadIdx.x;
    const int dim = blockIdx.y;

    if(idx >= N) {
        return;
    }

    if(du_dx) {

        double f = f_buffer[0];
        double g = g_buffer[0];
        double h = h_buffer[0];

        double df = df_dx_buffer[idx*3+dim];
        double dg = dg_dx_buffer[idx*3+dim];
        double dh = dh_dx_buffer[idx*3+dim];

        // (ytz): quotient rule
        double v = (2*f)/(g+h);
        double prefactor = 2*k*(v-1);
        double val = prefactor*2*((df*(g+h) - f*(dg+dh))/((g+h)*(g+h)));
        atomicAdd(du_dx + idx*3 + dim, static_cast<unsigned long long>((long long)(val*FIXED_EXPONENT)));
    }

}


void __global__ k_reduce_vol_buffer(
    const double *f_buf,
    const double *g_buf,
    const double *h_buf,
    const double k,
    double *u) {

    if(threadIdx.x > 0) {
        return;
    } 

    if(u) {
        double v = 2*f_buf[0]/(g_buf[0] + h_buf[0]);
        atomicAdd(u, k*(v-1)*(v-1));
    }

}

template<typename RealType>
void __global__ k_compute_volume(
    const double *coords,
    const double *alphas,
    const double *weights,
    const int A,
    const int B,
    const int *a_idxs,
    const int *b_idxs,
    double *vol_buffer,
    double *dvol_dx_buffer) {

    // A threads going B times
    const int a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx >= A) {
        return;
    }

    const int i_idx = a_idxs[a_idx];
    const RealType ai = alphas[i_idx];
    const RealType pi = weights[i_idx];

    const RealType xi = coords[i_idx*3 + 0];
    const RealType yi = coords[i_idx*3 + 1];
    const RealType zi = coords[i_idx*3 + 2];

    RealType dxi = 0;
    RealType dyi = 0;
    RealType dzi = 0;

    double sum = 0;

    for(int b_idx = 0; b_idx < B; b_idx++) {
        const int j_idx = b_idxs[b_idx];
        RealType aj = alphas[j_idx];
        RealType pj = weights[j_idx];

        RealType xj = coords[j_idx*3 + 0];
        RealType yj = coords[j_idx*3 + 1];
        RealType zj = coords[j_idx*3 + 2];

        RealType dx = xi - xj;
        RealType dy = yi - yj;
        RealType dz = zi - zj;

        RealType d2ij = dx*dx + dy*dy + dz*dz;

        RealType kij = exp(-(ai*aj*d2ij)/(ai+aj));
        RealType vij = pi*pj*kij*pow(PI/(ai+aj), 1.5);
        RealType dij = sqrt(d2ij); // this can be cancelled out of the dkij_dr

        RealType dkij_dr = -((ai*aj)/(ai+aj))*2*kij;
        RealType dvij_dr = pi*pj*dkij_dr*pow(PI/(ai+aj), 1.5);

        dxi += dvij_dr*dx;
        dyi += dvij_dr*dy;
        dzi += dvij_dr*dz;

        atomicAdd(dvol_dx_buffer + j_idx*3 + 0, -dvij_dr*dx);
        atomicAdd(dvol_dx_buffer + j_idx*3 + 1, -dvij_dr*dy);
        atomicAdd(dvol_dx_buffer + j_idx*3 + 2, -dvij_dr*dz);

        sum += vij;

    }

    atomicAdd(dvol_dx_buffer + i_idx*3 + 0, dxi);
    atomicAdd(dvol_dx_buffer + i_idx*3 + 1, dyi);
    atomicAdd(dvol_dx_buffer + i_idx*3 + 2, dzi);

    atomicAdd(vol_buffer, sum);
}

template <typename RealType>
Shape<RealType>::Shape(
    const int N,
    const std::vector<int> &a_idxs,
    const std::vector<int> &b_idxs,
    const std::vector<double> &alphas,
    const std::vector<double> &weights,
    double k
) : N_(N), A_(a_idxs.size()), B_(b_idxs.size()), k_(k) {

    for(auto a: a_idxs) {
        if(a >= N) {
            throw std::runtime_error("Shape::Shape() bad a in a_idxs()");
        }
    }

    for(auto b: b_idxs) {
        if(b >= N) {
            throw std::runtime_error("Shape::Shape() bad a in a_idxs()");
        }
    }

    if(alphas.size() != N) {
        throw std::runtime_error("Shape::Shape() bad alphas size != N");
    }

    if(weights.size() != N) {
        throw std::runtime_error("Shape::Shape() bad weights size != N");
    }

    gpuErrchk(cudaMalloc(&d_a_idxs_, A_*sizeof(*d_a_idxs_)));
    gpuErrchk(cudaMemcpy(d_a_idxs_, &a_idxs[0], A_*sizeof(*d_a_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_b_idxs_, B_*sizeof(*d_b_idxs_)));
    gpuErrchk(cudaMemcpy(d_b_idxs_, &b_idxs[0], B_*sizeof(*d_b_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_weights_, N_*sizeof(*d_weights_)));
    gpuErrchk(cudaMemcpy(d_weights_, &weights[0], N_*sizeof(*d_weights_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_alphas_, N_*sizeof(*d_alphas_)));
    gpuErrchk(cudaMemcpy(d_alphas_, &alphas[0], N_*sizeof(*d_alphas_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_f_buffer_, sizeof(*d_f_buffer_)));
    gpuErrchk(cudaMalloc(&d_g_buffer_, sizeof(*d_g_buffer_)));
    gpuErrchk(cudaMalloc(&d_h_buffer_, sizeof(*d_h_buffer_)));

    gpuErrchk(cudaMalloc(&d_df_dx_buffer_, N_*3*sizeof(*d_df_dx_buffer_)));
    gpuErrchk(cudaMalloc(&d_dg_dx_buffer_, N_*3*sizeof(*d_dg_dx_buffer_)));
    gpuErrchk(cudaMalloc(&d_dh_dx_buffer_, N_*3*sizeof(*d_dh_dx_buffer_)));

};

template <typename RealType>
Shape<RealType>::~Shape() {
    gpuErrchk(cudaFree(d_a_idxs_));
    gpuErrchk(cudaFree(d_b_idxs_));
    gpuErrchk(cudaFree(d_f_buffer_));
    gpuErrchk(cudaFree(d_g_buffer_));
    gpuErrchk(cudaFree(d_h_buffer_));
    gpuErrchk(cudaFree(d_df_dx_buffer_));
    gpuErrchk(cudaFree(d_dg_dx_buffer_));
    gpuErrchk(cudaFree(d_dh_dx_buffer_));
    gpuErrchk(cudaFree(d_alphas_));
    gpuErrchk(cudaFree(d_weights_));
};

template <typename RealType>
void Shape<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    double *d_du_dl,
    double *d_u,
    cudaStream_t stream) {

    if(N != N_) {
        throw std::runtime_error("Shape::execute_device() N != N_");
    }

    int tpb = 32;

    gpuErrchk(cudaMemset(d_f_buffer_, 0, sizeof(*d_f_buffer_)));
    gpuErrchk(cudaMemset(d_g_buffer_, 0, sizeof(*d_g_buffer_)));
    gpuErrchk(cudaMemset(d_h_buffer_, 0, sizeof(*d_h_buffer_)));

    gpuErrchk(cudaMemset(d_df_dx_buffer_, 0, N*3*sizeof(*d_f_buffer_)));
    gpuErrchk(cudaMemset(d_dg_dx_buffer_, 0, N*3*sizeof(*d_g_buffer_)));
    gpuErrchk(cudaMemset(d_dh_dx_buffer_, 0, N*3*sizeof(*d_h_buffer_)));

    k_compute_volume<RealType><<<(B_+tpb-1)/tpb, tpb, 0, stream>>>(
        d_x,
        d_alphas_,
        d_weights_,
        A_,
        B_,
        d_a_idxs_,
        d_b_idxs_,
        d_f_buffer_,
        d_df_dx_buffer_
    );

    gpuErrchk(cudaPeekAtLastError());

    k_compute_volume<RealType><<<(A_+tpb-1)/tpb, tpb, 0, stream>>>(
        d_x,
        d_alphas_,
        d_weights_,
        A_,
        A_,
        d_a_idxs_,
        d_a_idxs_,
        d_g_buffer_,
        d_dg_dx_buffer_
    );

    gpuErrchk(cudaPeekAtLastError());

    k_compute_volume<RealType><<<(B_+tpb-1)/tpb, tpb, 0, stream>>>(
        d_x,
        d_alphas_,
        d_weights_,
        B_,
        B_,
        d_b_idxs_,
        d_b_idxs_,
        d_h_buffer_,
        d_dh_dx_buffer_
    );

    gpuErrchk(cudaPeekAtLastError());

    k_reduce_vol_buffer<<<1, 32, 0, stream>>>(
        d_f_buffer_,
        d_g_buffer_,
        d_h_buffer_,
        k_,
        d_u
    );

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid((N_+tpb-1)/tpb, 3, 1);

    k_reduce_dvol_dx_buffers<<<dimGrid, tpb, 0, stream>>>(
        N,
        d_f_buffer_,
        d_df_dx_buffer_,
        d_g_buffer_,
        d_dg_dx_buffer_,
        d_h_buffer_,
        d_dh_dx_buffer_,
        k_,
        d_du_dx
    );

    gpuErrchk(cudaPeekAtLastError());

};

template class Shape<double>;
template class Shape<float>;

} // namespace jankmachine