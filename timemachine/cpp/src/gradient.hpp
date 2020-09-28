// #pragma once

// #include <cuda_runtime.h>

// namespace timemachine {

// class Params {

//     int P_;

//     double* d_params_; // [P,] immutable
//     double* d_avg_du_dp_; // [P, ]accumulator

// public:

//     Params(
//         int P,
//         double *h_params
//     );

//     ~Params();

// }

// class Potential {

// public: 

//     virtual ~Gradient(parameters);

//     void execute_host(
//         const int N,
//         const double *h_coords,
//         const double *h_params,
//         const double *h_box,
//         const double lambda,
//         unsigned long long *h_du_dx,
//         double *h_du_dp,
//         double *h_du_dl,
//         double *h_u
//     );

//     virtual void execute_device(
//         const int N,
//         const double *d_coords,
//         const double *d_params,
//         const double *d_box,
//         const double lambda,
//         unsigned long long *d_du_dx,
//         double *d_du_dp,
//         double *d_du_dl,
//         double *d_u
//     ) = 0;



//     // void execute_lambda_jvp_host(
//     //     const int N,
//     //     const double *h_coords_primals,
//     //     const double *h_coords_tangents,
//     //     const double lambda_primal,
//     //     const double lambda_tangent,
//     //     double *h_out_coords_primals,
//     //     double *h_out_coords_tangents
//     // );

//     // virtual void execute_lambda_inference_device(
//     //     const int N,
//     //     const double *d_coords_primals,
//     //     const double lambda_primal,
//     //     unsigned long long *d_out_coords_primals,
//     //     double *d_out_lambda_primals,
//     //     double *d_out_energy_primal,
//     //     cudaStream_t stream
//     // ) = 0;

//     // virtual void execute_lambda_jvp_device(
//     //     const int N,
//     //     const double *d_coords_primals,
//     //     const double *d_coords_tangents,
//     //     const double lambda_primal,
//     //     const double lambda_tangent,
//     //     double *d_out_coords_primals,
//     //     double *d_out_coords_tangents,
//     //     cudaStream_t stream
//     // ) = 0;



// };

// }
