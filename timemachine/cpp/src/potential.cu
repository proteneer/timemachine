#include <memory>

#include "device_buffer.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "potential.hpp"

#include <chrono>
#include <iostream>

namespace timemachine {

const int Potential::D = 3;

void Potential::execute_batch_host(
    const int coord_batch_size,  // Number of batches of coordinates
    const int N,                 // Number of atoms
    const int param_batch_size,  // Number of batches of parameters
    const int P,                 // Number of parameters
    const int lambda_batch_size, // Number of lambda values
    const double *h_x,           // [coord_batch_size, N, 3]
    const double *h_p,           // [param_batch_size, P]
    const double *h_box,         // [coord_batch_size, 3, 3]
    const double *lambdas,       // [lambda_batch_size]
    unsigned long long *h_du_dx, // [coord_batch_size, param_batch_size, lambda_batch_size, N, 3]
    unsigned long long *h_du_dp, // [coord_batch_size, param_batch_size, lambda_batch_size, P]
    unsigned long long *h_du_dl, // [coord_batch_size, param_batch_size, lambda_batch_size, N]
    unsigned long long *h_u) {   // [coord_batch_size, param_batch_size, lambda_batch_size, N]
    std::unique_ptr<DeviceBuffer<double>> d_p(nullptr);
    if (P > 0) {
        d_p.reset(new DeviceBuffer<double>(param_batch_size * P));
        d_p->copy_from(h_p);
    }

    DeviceBuffer<double> d_box(coord_batch_size * D * D);
    d_box.copy_from(h_box);

    DeviceBuffer<double> d_x_buffer(coord_batch_size * N * D);
    d_x_buffer.copy_from(h_x);

    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dx_buffer(nullptr);
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dp_buffer(nullptr);
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dl_buffer(nullptr);
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_u_buffer(nullptr);

    const int total_executions = coord_batch_size * param_batch_size * lambda_batch_size;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    if (h_du_dx) {
        d_du_dx_buffer.reset(new DeviceBuffer<unsigned long long>(total_executions * N * D));
        gpuErrchk(cudaMemsetAsync(d_du_dx_buffer->data, 0, d_du_dx_buffer->size, stream));
    }

    if (h_du_dp) {
        d_du_dp_buffer.reset(new DeviceBuffer<unsigned long long>(total_executions * P));
        gpuErrchk(cudaMemsetAsync(d_du_dp_buffer->data, 0, d_du_dp_buffer->size, stream));
    }

    if (h_du_dl) {
        d_du_dl_buffer.reset(new DeviceBuffer<unsigned long long>(total_executions * N));
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer->data, 0, d_du_dl_buffer->size, stream));
    }

    if (h_u) {
        d_u_buffer.reset(new DeviceBuffer<unsigned long long>(total_executions * N));
        gpuErrchk(cudaMemsetAsync(d_u_buffer->data, 0, d_u_buffer->size, stream));
    }

    for (unsigned int i = 0; i < coord_batch_size; i++) {
        for (unsigned int j = 0; j < param_batch_size; j++) {
            for (unsigned int k = 0; k < lambda_batch_size; k++) {
                unsigned int offset_factor = (i * param_batch_size * lambda_batch_size) + (j * lambda_batch_size) + k;
                this->execute_device(
                    N,
                    P,
                    d_x_buffer.data + (i * N * D),
                    P > 0 ? d_p->data + (j * P) : nullptr,
                    d_box.data + (i * D * D),
                    lambdas[k],
                    d_du_dx_buffer ? d_du_dx_buffer->data + (offset_factor * N * D) : nullptr,
                    d_du_dp_buffer ? d_du_dp_buffer->data + (offset_factor * P) : nullptr,
                    d_du_dl_buffer ? d_du_dl_buffer->data + (offset_factor * N) : nullptr,
                    d_u_buffer ? d_u_buffer->data + (offset_factor * N) : nullptr,
                    stream);
            }
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));

    if (h_du_dx) {
        d_du_dx_buffer->copy_to(h_du_dx);
    }

    if (h_du_dp) {
        d_du_dp_buffer->copy_to(h_du_dp);
    }

    if (h_du_dl) {
        d_du_dl_buffer->copy_to(h_du_dl);
    }

    if (h_u) {
        d_u_buffer->copy_to(h_u);
    }
}

void Potential::execute_host(
    const int N,
    const int P,
    const double *h_x,           // [N,3]
    const double *h_p,           // [P,]
    const double *h_box,         // [3, 3]
    const double lambda,         // [1]
    unsigned long long *h_du_dx, // [N,3]
    unsigned long long *h_du_dp, // [P]
    unsigned long long *h_du_dl, //
    unsigned long long *h_u) {

    const int &D = Potential::D;

    DeviceBuffer<double> d_x(N * D);
    DeviceBuffer<double> d_box(D * D);

    d_x.copy_from(h_x);
    d_box.copy_from(h_box);

    std::unique_ptr<DeviceBuffer<double>> d_p;
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dx;
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dp;
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_du_dl;
    std::unique_ptr<DeviceBuffer<unsigned long long>> d_u;

    if (P > 0) {
        d_p.reset(new DeviceBuffer<double>(P));
        d_p->copy_from(h_p);
    }

    // very important that these are initialized to zero since the kernels themselves just accumulate
    if (h_du_dx) {
        d_du_dx.reset(new DeviceBuffer<unsigned long long>(N * D));
        gpuErrchk(cudaMemset(d_du_dx->data, 0, d_du_dx->size));
    }
    if (h_du_dp) {
        d_du_dp.reset(new DeviceBuffer<unsigned long long>(P));
        gpuErrchk(cudaMemset(d_du_dp->data, 0, d_du_dp->size));
    }
    if (h_du_dl) {
        d_du_dl.reset(new DeviceBuffer<unsigned long long>(N));
        gpuErrchk(cudaMemset(d_du_dl->data, 0, d_du_dl->size));
    }
    if (h_u) {
        d_u.reset(new DeviceBuffer<unsigned long long>(N));
        gpuErrchk(cudaMemset(d_u->data, 0, d_u->size));
    }

    this->execute_device(
        N,
        P,
        d_x.data,
        P > 0 ? d_p->data : nullptr,
        d_box.data,
        lambda,
        d_du_dx ? d_du_dx->data : nullptr,
        d_du_dp ? d_du_dp->data : nullptr,
        d_du_dl ? d_du_dl->data : nullptr,
        d_u ? d_u->data : nullptr,
        static_cast<cudaStream_t>(0));

    // outputs
    if (h_du_dx) {
        d_du_dx->copy_to(h_du_dx);
    }
    if (h_du_dp) {
        d_du_dp->copy_to(h_du_dp);
    }
    if (h_du_dl) {
        d_du_dl->copy_to(h_du_dl);
    }
    if (h_u) {
        d_u->copy_to(h_u);
    }
};

void Potential::execute_host_du_dx(
    const int N,
    const int P,
    const double *h_x,   // [N,3]
    const double *h_p,   // [P,]
    const double *h_box, // [3, 3]
    const double lambda, // [1]
    unsigned long long *h_du_dx) {

    const int &D = Potential::D;

    double *d_x;
    double *d_p;
    double *d_box;

    gpuErrchk(cudaMalloc(&d_x, N * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N * D * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_p, P * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D * D * sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx; // du/dx

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_du_dx, N * D * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dx, 0, N * D * sizeof(unsigned long long)));

    this->execute_device(
        N, P, d_x, d_p, d_box, lambda, d_du_dx, nullptr, nullptr, nullptr, static_cast<cudaStream_t>(0));

    gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N * D * sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dx));
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_box));
};

void Potential::du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {
    for (int i = 0; i < P; i++) {
        du_dp_float[i] = FIXED_TO_FLOAT<double>(du_dp[i]);
    }
}

} // namespace timemachine
