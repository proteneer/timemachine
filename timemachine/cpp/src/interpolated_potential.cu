#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "interpolated_potential.hpp"

#include <cub/cub.cuh>

#include <stdexcept>

namespace timemachine {

__global__ void k_final_add(const double *in, double *out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1) {
        return;
    }

    atomicAdd(out, in[0]);
}

__global__ void k_reduce_du_dl(
    const double lambda,
    const int P_base,
    const double *d_du_dp_buf, // interpolated
    const double *d_p_src,
    const double *d_p_dst,
    double *d_du_dl_buf) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= P_base) {
        return;
    }

    d_du_dl_buf[idx] = d_du_dp_buf[idx] * (d_p_dst[idx] - d_p_src[idx]);
}

__global__ void k_reduce_du_dp(
    const double lambda,
    const int P_base,
    const double *d_du_dp_buf, // interpolated
    double *d_du_dp) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= P_base) {
        return;
    }

    // no atomics needed since they write uniquely. unlike other terms (u, du_dx, du_dl),
    // the parameters are local and not shared between classes.
    d_du_dp[idx] += d_du_dp_buf[idx] * (1 - lambda);
    d_du_dp[P_base + idx] += d_du_dp_buf[idx] * lambda;
}

__global__ void k_interpolate_parameters(
    const double lambda,
    const int P_base,
    const double *__restrict__ params_src,
    const double *__restrict__ params_dst,
    double *params_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= P_base) {
        return;
    }

    params_out[idx] = (1 - lambda) * params_src[idx] + lambda * params_dst[idx];
}

InterpolatedPotential::InterpolatedPotential(std::shared_ptr<Potential> u, int N, int P) : u_(u) {

    if (P % 2 != 0) {
        throw std::runtime_error("P must be divisible by 2 for interpolation.");
    }

    int P_base = P / 2;

    gpuErrchk(cudaMalloc(&d_du_dp_buffer_, P * sizeof(*d_du_dp_buffer_)));
    // this has shape P_base because we need to dot over du_dp
    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, P_base * sizeof(*d_du_dp_buffer_)));
    gpuErrchk(cudaMalloc(&d_p_interpolated_, P_base * sizeof(*d_p_interpolated_)));
    gpuErrchk(cudaMalloc(&d_sum_storage_out_, sizeof(*d_sum_storage_out_)));

    // (ytz): pseudo-associative reduction, results may differ on different devices.
    // however they are consisten within the same device. an extra storage_out is required
    // because we cannot directly write out to d_du_dl since its an assignment and not
    // atomicAdd.
    cub::DeviceReduce::Sum(
        nullptr,              // void *d_temp_storage,
        d_sum_storage_bytes_, // size_t &temp_storage_bytes,
        d_du_dl_buffer_,      //InputIteratorT d_in,
        d_sum_storage_out_,   //OutputIteratorT d_out,
        P_base                // int num_items,
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sum_storage_buffer_, d_sum_storage_bytes_));
}

InterpolatedPotential::~InterpolatedPotential() {
    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_du_dl_buffer_)) gpuErrchk(cudaFree(d_p_interpolated_));
    gpuErrchk(cudaFree(d_sum_storage_buffer_));
    gpuErrchk(cudaFree(d_sum_storage_out_));
}

void InterpolatedPotential::execute_device(
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

    if (P % 2 != 0) {
        throw std::runtime_error("P must be divisible by 2 for interpolation.");
    }

    int P_base = P / 2;
    int tpb = 32;
    int B = (P_base + tpb - 1) / tpb;

    // clear buffers
    if (d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, P_base * sizeof(*d_du_dl_buffer_), stream));
    }

    if (d_du_dp || d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_du_dp_buffer_, 0, P * sizeof(*d_du_dp_buffer_), stream));
    }

    k_interpolate_parameters<<<B, tpb, 0, stream>>>(lambda, P_base, d_p, d_p + P_base, d_p_interpolated_);
    gpuErrchk(cudaPeekAtLastError());

    u_->execute_device(
        N,
        P / 2,
        d_x,
        d_p_interpolated_,
        d_box,
        lambda,
        d_du_dx,                                          // no buffering needed
        (d_du_dp || d_du_dl) ? d_du_dp_buffer_ : nullptr, // du_dl requires du_dp
        d_du_dl,
        d_u, // no buffering needed
        stream);

    if (d_du_dl) {

        // why is this zero for nonbonded terms? clearly charges are different!
        k_reduce_du_dl<<<B, tpb, 0, stream>>>(lambda, P_base, d_du_dp_buffer_, d_p, d_p + P_base, d_du_dl_buffer_);
        gpuErrchk(cudaPeekAtLastError());

        cub::DeviceReduce::Sum(
            d_sum_storage_buffer_, // void *d_temp_storage,
            d_sum_storage_bytes_,  // size_t &temp_storage_bytes,
            d_du_dl_buffer_,       //InputIteratorT d_in,
            d_sum_storage_out_,    //OutputIteratorT d_out,
            P_base                 // int num_items,
        );
        gpuErrchk(cudaPeekAtLastError());

        k_final_add<<<1, tpb, 0, stream>>>(d_sum_storage_out_, d_du_dl);
        gpuErrchk(cudaPeekAtLastError());
    }

    if (d_du_dp) {
        // we don't need a pseudo-associate reduction here because we write to unique indices.
        // only one instance of force should ever be present.
        k_reduce_du_dp<<<B, tpb, 0, stream>>>(lambda, P_base, d_du_dp_buffer_, d_du_dp);
        gpuErrchk(cudaPeekAtLastError());
    }
}

} // namespace timemachine
