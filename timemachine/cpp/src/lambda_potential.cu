
#include "lambda_potential.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

__global__ void k_reduce_add_force_buffer(
    int count,
    unsigned long long *out,
    unsigned long long *buffer,
    double lambda) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= count) {
        return;
    }

    // convert fixed to double
    double val = FIXED_TO_FLOAT<double>(buffer[idx]);
    val *= lambda;

    unsigned long long fixed_val = static_cast<unsigned long long>((long long) (val*FIXED_EXPONENT));

    atomicAdd(out + idx, fixed_val);

}


template<typename RealType>
__global__ void k_reduce_add_buffer(
    int count,
    RealType *out,
    RealType *buffer,
    double lambda) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= count) {
        return;
    }

    // does this work if one of them is an unsigned long long?
    atomicAdd(out + idx, lambda*buffer[idx]);

}

__global__ void k_reduce_add_du_dl(
    double *du_dl,
    double *u_buf,
    double *du_dl_buf,
    double multiplier,
    double offset,
    double lambda) {

    if(threadIdx.x == 0) {
        atomicAdd(du_dl, (multiplier*u_buf[0] + (multiplier*lambda + offset)*du_dl_buf[0]));
    }

}


LambdaPotential::LambdaPotential(
    std::shared_ptr<Potential> u,
    int N,
    int P,
    double multiplier,
    double offset) :
        u_(u), multiplier_(multiplier), offset_(offset) {

    gpuErrchk(cudaMalloc(&d_du_dx_buffer_, N*3*sizeof(*d_du_dx_buffer_)));
    gpuErrchk(cudaMalloc(&d_du_dp_buffer_, P*sizeof(*d_du_dp_buffer_)));
    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, 1*sizeof(*d_du_dl_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, 1*sizeof(*d_u_buffer_)));

}

LambdaPotential::~LambdaPotential() {
    gpuErrchk(cudaFree(d_du_dx_buffer_));
    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
}

void LambdaPotential::execute_device(
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

    // initialize buffer streams

    if(d_du_dx) {
       gpuErrchk(cudaMemsetAsync(d_du_dx_buffer_, 0, N*3*sizeof(*d_du_dx_buffer_), stream))
    }

    if(d_du_dp) {
       gpuErrchk(cudaMemsetAsync(d_du_dp_buffer_, 0, P*sizeof(*d_du_dp_buffer_), stream))
    }

    if(d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, 1*sizeof(*d_du_dl_buffer_)));
    }

    // extra check is due to the chain rule
    if(d_u || d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, 1*sizeof(*d_u_buffer_)));
    }

    // let the alchemical potential a(l) be:
    // a(l) = l*u(l)
    // da/dl = u + l*du/dl
    u_->execute_device(
        N,
        P,
        d_x,
        d_p,
        d_box,
        lambda,
        d_du_dx ? d_du_dx_buffer_ : nullptr,
        d_du_dp ? d_du_dp_buffer_ : nullptr,
        d_du_dl ? d_du_dl_buffer_ : nullptr,
        (d_du_dl || d_u) ? d_u_buffer_ : nullptr,
        stream
    );

    int tpb = 32;

    if(d_du_dx) {
        int count = N*3;
        int blocks = (count + tpb - 1)/tpb;
        k_reduce_add_force_buffer<<<blocks, tpb, 0, stream>>>(count, d_du_dx, d_du_dx_buffer_, multiplier_*lambda + offset_);
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_du_dp) {
        int blocks = (P + tpb - 1)/tpb;
        k_reduce_add_buffer<<<blocks, tpb, 0, stream>>>(P, d_du_dp, d_du_dp_buffer_, multiplier_*lambda + offset_);
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_du_dl) {
        k_reduce_add_du_dl<<<1, tpb, 0, stream>>>(d_du_dl, d_u_buffer_, d_du_dl_buffer_, multiplier_, offset_, lambda);
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_u) {
        k_reduce_add_buffer<<<1, tpb, 0, stream>>>(1, d_u, d_u_buffer_, multiplier_*lambda + offset_);
        gpuErrchk(cudaPeekAtLastError());
    }

}

}
