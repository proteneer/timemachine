#include "kernel_utils.cuh"
#include "optimizer.hpp"

namespace timemachine {

template <typename RealType>
void Optimizer<RealType>::step_host(
    const int N,
    const int D,
    const int P,
    const RealType *h_dE_dx,
    const RealType *h_d2E_dx2,
    const RealType *h_d2E_dxdp, // this is modified in place
    RealType *h_x_t,            // mutable
    RealType *h_v_t,            // mutable
    RealType *h_dx_dp_t,        // mutable
    RealType *h_dv_dp_t,        // mutable
    const RealType *h_noise_buffer) const {

    RealType *d_dE_dx;
    gpuErrchk(cudaMalloc((void **)&d_dE_dx, N * D * sizeof(*d_dE_dx)));
    RealType *d_d2E_dx2;
    gpuErrchk(cudaMalloc((void **)&d_d2E_dx2, N * N * D * D * sizeof(*d_d2E_dx2)));
    RealType *d_d2E_dxdp;
    gpuErrchk(cudaMalloc((void **)&d_d2E_dxdp, P * N * D * sizeof(*d_d2E_dxdp)));
    RealType *d_x_t;
    gpuErrchk(cudaMalloc((void **)&d_x_t, N * D * sizeof(*d_x_t)));
    RealType *d_v_t;
    gpuErrchk(cudaMalloc((void **)&d_v_t, N * D * sizeof(*d_v_t)));
    RealType *d_dx_dp_t;
    gpuErrchk(cudaMalloc((void **)&d_dx_dp_t, P * N * D * sizeof(*d_dx_dp_t)));
    RealType *d_dv_dp_t;
    gpuErrchk(cudaMalloc((void **)&d_dv_dp_t, P * N * D * sizeof(*d_dv_dp_t)));
    RealType *d_noise_buffer;
    gpuErrchk(cudaMalloc((void **)&d_noise_buffer, N * D * sizeof(*d_noise_buffer)));

    gpuErrchk(cudaMemcpy(d_x_t, h_x_t, N * D * sizeof(*d_x_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v_t, h_v_t, N * D * sizeof(*d_v_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dx_dp_t, h_dx_dp_t, P * N * D * sizeof(*d_dx_dp_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dv_dp_t, h_dv_dp_t, P * N * D * sizeof(*d_dv_dp_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_dE_dx, h_dE_dx, N * D * sizeof(*h_dE_dx), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_d2E_dx2, h_d2E_dx2, N * N * D * D * sizeof(*d_d2E_dx2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_d2E_dxdp, h_d2E_dxdp, P * N * D * sizeof(*d_d2E_dxdp), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_noise_buffer, h_noise_buffer, N * D * sizeof(*d_noise_buffer), cudaMemcpyHostToDevice));

    this->step(N, D, P, d_dE_dx, d_d2E_dx2, d_d2E_dxdp, d_x_t, d_v_t, d_dx_dp_t, d_dv_dp_t, d_noise_buffer);

    gpuErrchk(cudaMemcpy(h_x_t, d_x_t, N * D * sizeof(*d_x_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_v_t, d_v_t, N * D * sizeof(*d_v_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_dx_dp_t, d_dx_dp_t, P * N * D * sizeof(*d_dx_dp_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_dv_dp_t, d_dv_dp_t, P * N * D * sizeof(*d_dv_dp_t), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_dE_dx));
    gpuErrchk(cudaFree(d_d2E_dx2));
    gpuErrchk(cudaFree(d_d2E_dxdp));
    gpuErrchk(cudaFree(d_x_t));
    gpuErrchk(cudaFree(d_v_t));
    gpuErrchk(cudaFree(d_dx_dp_t));
    gpuErrchk(cudaFree(d_dv_dp_t));
    gpuErrchk(cudaFree(d_noise_buffer));
}

template class Optimizer<float>;
template class Optimizer<double>;

} // namespace timemachine
