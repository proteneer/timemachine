#include "fixed_point.hpp"
#include "alchemical_gradient.hpp"
#include "gpu_utils.cuh"

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
    gpuErrchk(cudaMalloc(&d_out_coords_tangents_buffer_u0_, N*D*sizeof(*d_out_coords_primals_buffer_u0_)));
    gpuErrchk(cudaMalloc(&d_out_params_tangents_buffer_u0_, P*sizeof(*d_out_energy_primal_buffer_u0_)));

    gpuErrchk(cudaMalloc(&d_out_coords_primals_buffer_u1_, N*D*sizeof(*d_out_coords_primals_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_energy_primal_buffer_u1_, sizeof(*d_out_energy_primal_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_lambda_primal_buffer_u1_, sizeof(*d_out_lambda_primal_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_coords_tangents_buffer_u1_, N*D*sizeof(*d_out_coords_primals_buffer_u1_)));
    gpuErrchk(cudaMalloc(&d_out_params_tangents_buffer_u1_, P*sizeof(*d_out_energy_primal_buffer_u1_)));

}

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
        *uc_energy = (*u0_energy)*(1-lambda) + (*u1_energy)*lambda;
        *uc_du_dl = -(*u0_energy) + (*u1_energy) + (*u0_du_dl)*(1-lambda) + (*u1_du_dl)*lambda;
    }

    const auto dim = blockIdx.y;

    if(idx < N) {
        auto f0 = static_cast<double>(static_cast<long long>(u0_coord_grads[idx*3+dim]))/FIXED_EXPONENT;
        auto f1 = static_cast<double>(static_cast<long long>(u1_coord_grads[idx*3+dim]))/FIXED_EXPONENT;
        auto fc = (1-lambda)*f0 + lambda*f1;
        uc_coord_grads[idx*3+dim] = static_cast<unsigned long long>((long long) (fc*FIXED_EXPONENT));
    }

}

void AlchemicalGradient::execute_lambda_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal,
    const double lambda_tangent,
    unsigned long long *d_out_coords_primals,
    double *d_out_lambda_primals,
    double *d_out_energy_primal,
    double *d_out_coords_tangents,
    double *d_out_params_tangents,
    cudaStream_t stream
) {

    // tbd cudaMemset
    const int D = 3;

    if(d_coords_tangents == nullptr) {

        gpuErrchk(cudaMemset(d_out_coords_primals_buffer_u0_, 0, N*D*sizeof(*d_out_coords_primals_buffer_u0_)));
        gpuErrchk(cudaMemset(d_out_lambda_primal_buffer_u0_, 0, sizeof(*d_out_lambda_primal_buffer_u0_)));
        gpuErrchk(cudaMemset(d_out_energy_primal_buffer_u0_, 0, sizeof(*d_out_energy_primal_buffer_u0_)));

        u0_->execute_lambda_device(
            N,
            P,
            d_coords_primals,
            d_coords_tangents,
            d_params_primals,
            lambda_primal,
            lambda_tangent,
            d_out_coords_primals_buffer_u0_,
            d_out_lambda_primal_buffer_u0_,
            d_out_energy_primal_buffer_u0_,
            nullptr,
            nullptr,
            stream
        );

        gpuErrchk(cudaMemset(d_out_coords_primals_buffer_u1_, 0, N*D*sizeof(*d_out_coords_primals_buffer_u1_)));
        gpuErrchk(cudaMemset(d_out_lambda_primal_buffer_u1_, 0, sizeof(*d_out_lambda_primal_buffer_u1_)));
        gpuErrchk(cudaMemset(d_out_energy_primal_buffer_u1_, 0, sizeof(*d_out_energy_primal_buffer_u1_)));

        u1_->execute_lambda_device(
            N,
            P,
            d_coords_primals,
            d_coords_tangents,
            d_params_primals,
            lambda_primal,
            lambda_tangent,
            d_out_coords_primals_buffer_u1_,
            d_out_lambda_primal_buffer_u1_,
            d_out_energy_primal_buffer_u1_,
            nullptr,
            nullptr,
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

}
