#include "device_buffer.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "potential.hpp"

namespace timemachine {

const int Potential::D = 3;

void Potential::execute_batch_device(
    const int coord_batch_size,
    const int N,
    const int param_batch_size,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    for (int i = 0; i < coord_batch_size; i++) {
        for (int j = 0; j < param_batch_size; j++) {
            unsigned int offset_factor = (i * param_batch_size) + j;
            this->execute_device(
                N,
                P,
                d_x + (i * N * D),
                P > 0 ? d_p + (j * P) : nullptr,
                d_box + (i * D * D),
                d_du_dx ? d_du_dx + (offset_factor * N * D) : nullptr,
                d_du_dp ? d_du_dp + (offset_factor * P) : nullptr,
                d_u ? d_u + offset_factor : nullptr,
                stream);
        }
    }
}

void Potential::execute_batch_sparse_device(
    const int N,
    const int P,
    const int batch_size,
    const unsigned int *coords_batch_idxs,
    const unsigned int *params_batch_idxs,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    for (int i = 0; i < batch_size; i++) {
        int ic = coords_batch_idxs[i];
        int ip = params_batch_idxs[i];
        this->execute_device(
            N,
            P,
            d_x + (ic * N * D),
            P > 0 ? d_p + (ip * P) : nullptr,
            d_box + (ic * D * D),
            d_du_dx ? d_du_dx + (i * N * D) : nullptr,
            d_du_dp ? d_du_dp + (i * P) : nullptr,
            d_u ? d_u + i : nullptr,
            stream);
    }
}

void Potential::execute_batch_host(
    const int coord_batch_size,  // Number of batches of coordinates
    const int N,                 // Number of atoms
    const int param_batch_size,  // Number of batches of parameters
    const int P,                 // Number of parameters
    const double *h_x,           // [coord_batch_size, N, 3]
    const double *h_p,           // [param_batch_size, P]
    const double *h_box,         // [coord_batch_size, 3, 3]
    unsigned long long *h_du_dx, // [coord_batch_size, param_batch_size, N, 3]
    unsigned long long *h_du_dp, // [coord_batch_size, param_batch_size, P]
    __int128 *h_u                // [coord_batch_size, param_batch_size]
) {
    DeviceBuffer<double> d_p(param_batch_size * P);
    if (P > 0) {
        d_p.copy_from(h_p);
    }

    DeviceBuffer<double> d_box(coord_batch_size * D * D);
    d_box.copy_from(h_box);

    DeviceBuffer<double> d_x_buffer(coord_batch_size * N * D);
    d_x_buffer.copy_from(h_x);

    DeviceBuffer<unsigned long long> d_du_dx_buffer;
    DeviceBuffer<unsigned long long> d_du_dp_buffer;
    DeviceBuffer<__int128> d_u_buffer;

    const int total_executions = coord_batch_size * param_batch_size;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    if (h_du_dx) {
        d_du_dx_buffer.realloc(total_executions * N * D);
        gpuErrchk(cudaMemsetAsync(d_du_dx_buffer.data, 0, d_du_dx_buffer.size(), stream));
    }

    if (h_du_dp) {
        d_du_dp_buffer.realloc(total_executions * P);
        gpuErrchk(cudaMemsetAsync(d_du_dp_buffer.data, 0, d_du_dp_buffer.size(), stream));
    }

    if (h_u) {
        d_u_buffer.realloc(total_executions);
        gpuErrchk(cudaMemsetAsync(d_u_buffer.data, 0, d_u_buffer.size(), stream));
    }

    this->execute_batch_device(
        coord_batch_size,
        N,
        param_batch_size,
        P,
        d_x_buffer.data,
        P > 0 ? d_p.data : nullptr,
        d_box.data,
        h_du_dx ? d_du_dx_buffer.data : nullptr,
        h_du_dp ? d_du_dp_buffer.data : nullptr,
        h_u ? d_u_buffer.data : nullptr,
        stream);

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));

    if (h_du_dx) {
        d_du_dx_buffer.copy_to(h_du_dx);
    }

    if (h_du_dp) {
        d_du_dp_buffer.copy_to(h_du_dp);
    }

    if (h_u) {
        d_u_buffer.copy_to(h_u);
    }
}

void Potential::execute_batch_sparse_host(
    const int coords_size,                 // Number of coordinate arrays
    const int N,                           // Number of atoms
    const int params_size,                 // Number of parameter arrays
    const int P,                           // Number of parameters
    const int batch_size,                  // Number of evaluations
    const unsigned int *coords_batch_idxs, // [batch_size] Index of the coordinates for each evaluation
    const unsigned int *params_batch_idxs, // [batch_size] Index of the parameters for each evaluation
    const double *h_x,                     // [coords_size, N, 3]
    const double *h_p,                     // [params_size, P]
    const double *h_box,                   // [coords_size, 3, 3]
    unsigned long long *h_du_dx,           // [batch_size, N, 3]
    unsigned long long *h_du_dp,           // [batch_size, P]
    __int128 *h_u                          // [batch_size]
) {
    DeviceBuffer<double> d_p(params_size * P);
    if (P > 0) {
        d_p.copy_from(h_p);
    }

    DeviceBuffer<double> d_box(coords_size * D * D);
    d_box.copy_from(h_box);

    DeviceBuffer<double> d_x_buffer(coords_size * N * D);
    d_x_buffer.copy_from(h_x);

    DeviceBuffer<unsigned long long> d_du_dx_buffer;
    DeviceBuffer<unsigned long long> d_du_dp_buffer;
    DeviceBuffer<__int128> d_u_buffer;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    if (h_du_dx) {
        d_du_dx_buffer.realloc(batch_size * N * D);
        gpuErrchk(cudaMemsetAsync(d_du_dx_buffer.data, 0, d_du_dx_buffer.size(), stream));
    }

    if (h_du_dp) {
        d_du_dp_buffer.realloc(batch_size * P);
        gpuErrchk(cudaMemsetAsync(d_du_dp_buffer.data, 0, d_du_dp_buffer.size(), stream));
    }

    if (h_u) {
        d_u_buffer.realloc(batch_size);
        gpuErrchk(cudaMemsetAsync(d_u_buffer.data, 0, d_u_buffer.size(), stream));
    }

    this->execute_batch_sparse_device(
        N,
        P,
        batch_size,
        coords_batch_idxs,
        params_batch_idxs,
        d_x_buffer.data,
        P > 0 ? d_p.data : nullptr,
        d_box.data,
        h_du_dx ? d_du_dx_buffer.data : nullptr,
        h_du_dp ? d_du_dp_buffer.data : nullptr,
        h_u ? d_u_buffer.data : nullptr,
        stream);

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));

    if (h_du_dx) {
        d_du_dx_buffer.copy_to(h_du_dx);
    }

    if (h_du_dp) {
        d_du_dp_buffer.copy_to(h_du_dp);
    }

    if (h_u) {
        d_u_buffer.copy_to(h_u);
    }
}

void Potential::execute_host(
    const int N,
    const int P,
    const double *h_x,           // [N,3]
    const double *h_p,           // [P,]
    const double *h_box,         // [3, 3]
    unsigned long long *h_du_dx, // [N,3]
    unsigned long long *h_du_dp, // [P]
    __int128 *h_u                // [1]
) {

    const int &D = Potential::D;

    DeviceBuffer<double> d_x(N * D);
    DeviceBuffer<double> d_box(D * D);

    d_x.copy_from(h_x);
    d_box.copy_from(h_box);

    DeviceBuffer<double> d_p(P);
    DeviceBuffer<unsigned long long> d_du_dx;
    DeviceBuffer<unsigned long long> d_du_dp;
    DeviceBuffer<__int128> d_u;

    // very important that these are initialized to zero since the kernels themselves just accumulate

    if (P > 0) {
        d_p.copy_from(h_p);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    // very important that these are initialized to zero since the kernels themselves just accumulate
    if (h_du_dx) {
        d_du_dx.realloc(N * D);
        gpuErrchk(cudaMemsetAsync(d_du_dx.data, 0, d_du_dx.size(), stream));
    }
    if (h_du_dp) {
        d_du_dp.realloc(P);
        gpuErrchk(cudaMemsetAsync(d_du_dp.data, 0, d_du_dp.size(), stream));
    }
    if (h_u) {
        d_u.realloc(1);
        gpuErrchk(cudaMemsetAsync(d_u.data, 0, d_u.size(), stream));
    }

    this->execute_device(
        N,
        P,
        d_x.data,
        P > 0 ? d_p.data : nullptr,
        d_box.data,
        d_du_dx.length > 0 ? d_du_dx.data : nullptr,
        d_du_dp.length > 0 ? d_du_dp.data : nullptr,
        d_u.length > 0 ? d_u.data : nullptr,
        stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    // outputs
    if (h_du_dx) {
        d_du_dx.copy_to(h_du_dx);
    }
    if (h_du_dp) {
        d_du_dp.copy_to(h_du_dp);
    }
    if (h_u) {
        d_u.copy_to(h_u);
    }
};

void Potential::execute_host_du_dx(
    const int N,
    const int P,
    const double *h_x,   // [N,3]
    const double *h_p,   // [P,]
    const double *h_box, // [3, 3]
    unsigned long long *h_du_dx) {

    const int &D = Potential::D;

    DeviceBuffer<double> d_x(N * D);
    DeviceBuffer<double> d_p(P);
    DeviceBuffer<double> d_box(D * D);

    d_x.copy_from(h_x);
    d_p.copy_from(h_p);
    d_box.copy_from(h_box);

    DeviceBuffer<unsigned long long> d_du_dx(N * D);

    gpuErrchk(cudaMemset(d_du_dx.data, 0, d_du_dx.size()));

    this->execute_device(
        N, P, d_x.data, d_p.data, d_box.data, d_du_dx.data, nullptr, nullptr, static_cast<cudaStream_t>(0));

    d_du_dx.copy_to(h_du_dx);
};

void Potential::reset() {};

void Potential::du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {
    for (int i = 0; i < P; i++) {
        du_dp_float[i] = FIXED_TO_FLOAT<double>(du_dp[i]);
    }
}

} // namespace timemachine
