#include <cuda_fp16.h>
#include "bitplay.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

BitPlay::BitPlay(
    int N,
    const double *x_0
    ){
    N_ = N;
    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N*3);
    gpuErrchk(cudaMalloc(&d_dist_, N*N*sizeof(*d_dist_)));

};

BitPlay::~BitPlay() {
    gpuErrchk(cudaFree(d_x_t_));
};

void __global__ fp16_compute_dist(const int N, const double* __restrict__ x_0, float* __restrict__ dists) {

    const int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index + 1 > N) {
        return;
    }

    __half x1 = __double2half(x_0[index*2+0]);
    __half y1 = __double2half(x_0[(index*2)+1]);
    __half x2 = __double2half(x_0[(index+1)*2+0]);
    __half y2 = __double2half(x_0[(index+1)*2+1]);
    __half2 x = __halves2half2(x1, x2);
    __half2 y = __halves2half2(y1, y2);
    for (int i = 0; i < N; i++) {
        __half comp_x = __double2half(x_0[i*2]);
        __half comp_y = __double2half(x_0[i*2+1]);
        __half2 combo_x = __half2half2(comp_x);
        __half2 combo_y = __half2half2(comp_y);

        __half2 delta_x = __hsub2(x, combo_x);
        __half2 delta_y = __hsub2(y, combo_y);
        __half2 sq_dist = __hadd2(__hmul2(delta_x, delta_x), __hmul2(delta_y, delta_y));

        __half2 dist = h2sqrt(sq_dist);
        dists[i*N+index] = __half2float(__low2half(dist));
        dists[i*N+(index+1)] = __half2float(__high2half(dist));
    }
}

void __global__ fp32_compute_dist(const int N, const double* __restrict__ x_0, float* __restrict__ dists) {

    const int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index + 1 > N) {
        return;
    }

    float x = x_0[index*2+0];
    float y = x_0[(index*2)+1];
    for (int i = 0; i < N; i++) {
        float comp_x = x_0[i*2];
        float comp_y = x_0[i*2+1];

        float delta_x = x - comp_x;
        float delta_y = y - comp_y;
        float sq_dist = delta_x*delta_x + delta_y*delta_y;

        float dist = sqrt(sq_dist);
        dists[i*N+index] = dist;
    }
}

void __global__ tensor_compute_dist(const int N, const double* __restrict__ x_0, half2* __restrict__ dist) {
    return;
}

void BitPlay::compute_dist() {
    const int tpb = 32;
    const int half_n = (N_ + 2 - 1) / 2;
    const int blocks = (half_n + tpb - 1) / tpb;
    fp16_compute_dist<<<blocks, tpb>>>(N_, d_x_t_, d_dist_);
}

void BitPlay::compute_dist_f32() {
    const int tpb = 32;
    const int blocks = (N_ + tpb - 1) / tpb;
    fp32_compute_dist<<<blocks, tpb>>>(N_, d_x_t_, d_dist_);
}

void BitPlay::get_dist(float *out_buffer) {
    gpuErrchk(cudaMemcpy(out_buffer, d_dist_, N_*N_*sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void BitPlay::compute_dist_tensor() {
    return;
}

}

