#include <iostream>

#include "fixed_point.hpp"
#include "alchemical_gradient.hpp"
#include "gpu_utils.cuh"
#include "surreal.cuh"

namespace timemachine {

AlchemicalGradient::AlchemicalGradient(
    const int N,
    const int P,
    Gradient *u0,
    Gradient *u1) : 
    u0_(u0),
    u1_(u1) {

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_out_coords_primals_buffer_u0_, N*D*sizeof(*d_out_coords_primals_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_energy_primal_buffer_u0_, sizeof(*d_out_energy_primal_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_lambda_primal_buffer_u0_, sizeof(*d_out_lambda_primal_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_coords_primals_buffer_u0_, N*D*sizeof(*d_out_jvp_coords_primals_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_coords_tangents_buffer_u0_, N*D*sizeof(*d_out_jvp_coords_primals_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_params_primals_buffer_u0_, P*sizeof(*d_out_jvp_params_primals_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_params_tangents_buffer_u0_, P*sizeof(*d_out_jvp_params_tangents_buffer_u0_)));

    gpuErrchk(cudaMalloc(&d_out_coords_primals_buffer_u1_, N*D*sizeof(*d_out_coords_primals_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_energy_primal_buffer_u1_, sizeof(*d_out_energy_primal_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_lambda_primal_buffer_u1_, sizeof(*d_out_lambda_primal_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_coords_primals_buffer_u1_, N*D*sizeof(*d_out_jvp_coords_primals_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_coords_tangents_buffer_u1_, N*D*sizeof(*d_out_jvp_coords_primals_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_params_primals_buffer_u1_, P*sizeof(*d_out_jvp_params_primals_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_jvp_params_tangents_buffer_u1_, P*sizeof(*d_out_jvp_params_tangents_buffer_u1_)));

}

AlchemicalGradient::~AlchemicalGradient() {
    gpuErrchk(cudaFree(d_out_coords_primals_buffer_u0_));
    gpuErrchk(cudaFree(d_out_energy_primal_buffer_u0_));
    gpuErrchk(cudaFree(d_out_lambda_primal_buffer_u0_));
    gpuErrchk(cudaFree(d_out_jvp_coords_primals_buffer_u0_));
    gpuErrchk(cudaFree(d_out_jvp_coords_tangents_buffer_u0_));
    gpuErrchk(cudaFree(d_out_jvp_params_primals_buffer_u0_));
    gpuErrchk(cudaFree(d_out_jvp_params_tangents_buffer_u0_));

    gpuErrchk(cudaFree(d_out_coords_primals_buffer_u1_));
    gpuErrchk(cudaFree(d_out_energy_primal_buffer_u1_));
    gpuErrchk(cudaFree(d_out_lambda_primal_buffer_u1_));
    gpuErrchk(cudaFree(d_out_jvp_coords_primals_buffer_u1_));
    gpuErrchk(cudaFree(d_out_jvp_coords_tangents_buffer_u1_));
    gpuErrchk(cudaFree(d_out_jvp_params_primals_buffer_u1_));
    gpuErrchk(cudaFree(d_out_jvp_params_tangents_buffer_u1_));

};

__global__ void k_linear_rescale_inference(
    double lambda,
    const int N,
    const unsigned long long *u0_coord_grads,
    const double *u0_du_dl,
    const double *u0_energy,
    const unsigned long long *u1_coord_grads,
    const double *u1_du_dl,
    const double *u1_energy,
    unsigned long long *uc_coord_grads,
    double *uc_du_dl,
    double *uc_energy) {

    const auto idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(idx == 0 && blockIdx.y == 0) {
        atomicAdd(uc_energy, (*u0_energy)*(1-lambda) + (*u1_energy)*lambda);
        atomicAdd(uc_du_dl, -(*u0_energy) + (*u1_energy) + (*u0_du_dl)*(1-lambda) + (*u1_du_dl)*lambda);
    }

    const auto dim = blockIdx.y;

    if(idx < N) {
        auto f0 = static_cast<double>(static_cast<long long>(u0_coord_grads[idx*3+dim]))/FIXED_EXPONENT;
        auto f1 = static_cast<double>(static_cast<long long>(u1_coord_grads[idx*3+dim]))/FIXED_EXPONENT;
        auto fc = (1-lambda)*f0 + lambda*f1;
        atomicAdd(uc_coord_grads + idx*3 + dim, static_cast<unsigned long long>((long long) (fc*FIXED_EXPONENT)));
    }

}

__global__ void k_linear_rescale_jvp(
    double lambda_primal,
    double lambda_tangent,
    const int N,
    const int P,
    const double *coords_primals_u0,
    const double *coords_tangents_u0,
    const double *params_primals_u0,
    const double *params_tangents_u0,

    const double *coords_primals_u1,
    const double *coords_tangents_u1,
    const double *params_primals_u1,
    const double *params_tangents_u1,

    double *coords_primals_uc,
    double *coords_tangents_uc,
    double *params_primals_uc,
    double *params_tangents_uc) {

    const auto idx = blockDim.x*blockIdx.x + threadIdx.x;
    const auto dim = blockIdx.y;

    Surreal<double> lambda(lambda_primal, lambda_tangent);

    if(idx < N) {
        Surreal<double> f0(coords_primals_u0[idx*3+dim], coords_tangents_u0[idx*3+dim]); 
        Surreal<double> f1(coords_primals_u1[idx*3+dim], coords_tangents_u1[idx*3+dim]); 
        auto fc = (1-lambda)*f0 + lambda*f1;
        atomicAdd(coords_primals_uc + idx*3+dim, fc.real);
        atomicAdd(coords_tangents_uc + idx*3+dim, fc.imag);
    }

    if(idx < P && blockIdx.y == 0) {
        Surreal<double> p0(params_primals_u0[idx], params_tangents_u0[idx]); 
        Surreal<double> p1(params_primals_u1[idx], params_tangents_u1[idx]);
        auto pc = (1-lambda)*p0 + lambda*p1;
        atomicAdd(params_primals_uc + idx, pc.real);
        atomicAdd(params_tangents_uc + idx, pc.imag);
    }

}

void AlchemicalGradient::execute_lambda_jvp_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal,
    const double lambda_tangent,
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    double *d_out_params_primals,
    double *d_out_params_tangents,
    cudaStream_t stream) {

    const int D = 3;

    gpuErrchk(cudaMemsetAsync(d_out_jvp_coords_primals_buffer_u0_, 0, N*D*sizeof(*d_out_jvp_coords_primals_buffer_u0_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_jvp_coords_tangents_buffer_u0_, 0, N*D*sizeof(*d_out_jvp_coords_tangents_buffer_u0_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_jvp_params_primals_buffer_u0_, 0, P*sizeof(*d_out_jvp_params_primals_buffer_u0_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_jvp_params_tangents_buffer_u0_, 0, P*sizeof(*d_out_jvp_params_tangents_buffer_u0_), stream));

    u0_->execute_lambda_jvp_device(
        N,
        P,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        lambda_primal,
        lambda_tangent,
        d_out_jvp_coords_primals_buffer_u0_,
        d_out_jvp_coords_tangents_buffer_u0_,
        d_out_jvp_params_primals_buffer_u0_,
        d_out_jvp_params_tangents_buffer_u0_,
        stream
    );

    gpuErrchk(cudaMemsetAsync(d_out_jvp_coords_primals_buffer_u1_, 0, N*D*sizeof(*d_out_jvp_coords_primals_buffer_u1_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_jvp_coords_tangents_buffer_u1_, 0, N*D*sizeof(*d_out_jvp_coords_tangents_buffer_u1_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_jvp_params_primals_buffer_u1_, 0, P*sizeof(*d_out_jvp_params_primals_buffer_u1_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_jvp_params_tangents_buffer_u1_, 0, P*sizeof(*d_out_jvp_params_tangents_buffer_u1_), stream));
    
    u1_->execute_lambda_jvp_device(
        N,
        P,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        lambda_primal,
        lambda_tangent,
        d_out_jvp_coords_primals_buffer_u1_,
        d_out_jvp_coords_tangents_buffer_u1_,
        d_out_jvp_params_primals_buffer_u1_,
        d_out_jvp_params_tangents_buffer_u1_,
        stream
    );

    int tpb = 32;
    int B = (max(N,P)+tpb-1)/tpb;
    dim3 dimGrid(B, 3, 1); // x, y, z dims

    k_linear_rescale_jvp<<<dimGrid, tpb, 0, stream>>>(
        lambda_primal,
        lambda_tangent,
        N,
        P,
        d_out_jvp_coords_primals_buffer_u0_,
        d_out_jvp_coords_tangents_buffer_u0_,
        d_out_jvp_params_primals_buffer_u0_,
        d_out_jvp_params_tangents_buffer_u0_,
        d_out_jvp_coords_primals_buffer_u1_,
        d_out_jvp_coords_tangents_buffer_u1_,
        d_out_jvp_params_primals_buffer_u1_,
        d_out_jvp_params_tangents_buffer_u1_,
        d_out_coords_primals,
        d_out_coords_tangents,
        d_out_params_primals,
        d_out_params_tangents
    );

    gpuErrchk(cudaPeekAtLastError());



}

void AlchemicalGradient::execute_lambda_inference_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_params_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals,
    double *d_out_lambda_primals,
    double *d_out_energy_primal,
    cudaStream_t stream) {

    // reset buffers
    const int D = 3;

    gpuErrchk(cudaMemsetAsync(d_out_coords_primals_buffer_u0_, 0, N*D*sizeof(*d_out_coords_primals_buffer_u0_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_lambda_primal_buffer_u0_, 0, sizeof(*d_out_lambda_primal_buffer_u0_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_energy_primal_buffer_u0_, 0, sizeof(*d_out_energy_primal_buffer_u0_), stream));
    u0_->execute_lambda_inference_device(
        N,
        P,
        d_coords_primals,
        d_params_primals,
        lambda_primal,
        d_out_coords_primals_buffer_u0_,
        d_out_lambda_primal_buffer_u0_,
        d_out_energy_primal_buffer_u0_,
        stream
    );

    gpuErrchk(cudaMemsetAsync(d_out_coords_primals_buffer_u1_, 0, N*D*sizeof(*d_out_coords_primals_buffer_u1_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_lambda_primal_buffer_u1_, 0, sizeof(*d_out_lambda_primal_buffer_u1_), stream));
    gpuErrchk(cudaMemsetAsync(d_out_energy_primal_buffer_u1_, 0, sizeof(*d_out_energy_primal_buffer_u1_), stream));
    u1_->execute_lambda_inference_device(
        N,
        P,
        d_coords_primals,
        d_params_primals,
        lambda_primal,
        d_out_coords_primals_buffer_u1_,
        d_out_lambda_primal_buffer_u1_,
        d_out_energy_primal_buffer_u1_,
        stream
    );

    int tpb = 32;
    int B = (N+tpb-1)/tpb;
    dim3 dimGrid(B, 3, 1); // x, y, z dims

    k_linear_rescale_inference<<<dimGrid, tpb, 0, stream>>>(
        lambda_primal,
        N,
        d_out_coords_primals_buffer_u0_,
        d_out_lambda_primal_buffer_u0_,
        d_out_energy_primal_buffer_u0_,
        d_out_coords_primals_buffer_u1_,
        d_out_lambda_primal_buffer_u1_,
        d_out_energy_primal_buffer_u1_,
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy_primal
    );

    gpuErrchk(cudaPeekAtLastError());

}

}
