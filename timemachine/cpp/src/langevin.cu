// #include "cublas_v2.h"
// #include "cusparse_v2.h"
// #include "curand.h"

// #include <iostream>
// #include <vector>
// #include <stdexcept>
// #include <cstdio>
// #include <chrono>

// #include "langevin.hpp"
// #include "gpu_utils.cuh"



// template <typename RealType>
// __global__ void update_positions(
//     const RealType *noise,
//     const RealType coeff_a,
//     const RealType *coeff_bs, // N x 3, not P x N x 3, but we could just pass in the first index
//     const RealType *coeff_cs,
//     const RealType coeff_d,
//     const RealType coeff_e,
//     const RealType *dE_dx,
//     const RealType dt,
//     const int N,
//     const int D,
//     RealType *x_t,
//     RealType *v_t) {


//     int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
//     if(atom_idx >= N) {
//         return;
//     }

//     int d_idx = blockIdx.y;


//     int local_idx = atom_idx*D + d_idx;
//     // only integrate first three dimensions
//     if(d_idx >= 3) {

//         // allow for growth in the higher dimensions
//         x_t[local_idx] *= coeff_e;
//         return;
//     }


//     // truncated noise
//     auto n = noise[local_idx];
//     v_t[local_idx] = coeff_a*v_t[local_idx] - coeff_bs[atom_idx]*dE_dx[local_idx] + coeff_cs[atom_idx]*n;
//     x_t[local_idx] = (1 - coeff_d*coeff_bs[atom_idx]*dt)*x_t[local_idx] + v_t[local_idx]*dt;

// }


// template<typename RealType>
// __global__ void update_derivatives(
//     const RealType coeff_a,
//     const RealType *coeff_bs, // shape N
//     const RealType coeff_d,
//     const RealType *d2E_dxdp, 
//     const RealType dt,
//     const int N,
//     const int D,
//     RealType *dx_dp_t,
//     RealType *dv_dp_t) {

//     int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
//     if(atom_idx >= N) {
//         return;
//     }

//     // only integrate first three dimensions
//     int d_idx = blockIdx.y;
//     if(d_idx >= 3) {
//         return;
//     }
// int p_idx = blockIdx.z;
//     int local_idx = p_idx*N*D + atom_idx*D + d_idx;

//     // derivative of the above equation
//     RealType tmp = coeff_a*dv_dp_t[local_idx] - coeff_bs[atom_idx]*d2E_dxdp[local_idx];
//     dv_dp_t[local_idx] = tmp;
//     dx_dp_t[local_idx] = (1 - coeff_d*coeff_bs[atom_idx]*dt)*dx_dp_t[local_idx] + dt*tmp;

// }


// namespace timemachine {


// template<typename RealType> 
// LangevinOptimizer<RealType>::LangevinOptimizer(
//     RealType dt,
//     const int num_dims,
//     const RealType coeff_a,
//     const std::vector<RealType> &coeff_bs,
//     const std::vector<RealType> &coeff_cs,
//     const int no) :
//     dt_(dt),
//     coeff_a_(coeff_a),
//     coeff_d_(0.0),
//     coeff_e_(1.0),
//     d_rng_buffer_(nullptr),
//     N_offset_(no) {

//     auto start = std::chrono::high_resolution_clock::now();
//     gpuErrchk(cudaMalloc((void**)&d_coeff_bs_, coeff_bs.size()*sizeof(RealType)));
//     gpuErrchk(cudaMalloc((void**)&d_coeff_cs_, coeff_cs.size()*sizeof(RealType)));

//     gpuErrchk(cudaMemcpy(d_coeff_bs_, &coeff_bs[0], coeff_bs.size()*sizeof(RealType), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_coeff_cs_, &coeff_cs[0], coeff_cs.size()*sizeof(RealType), cudaMemcpyHostToDevice));

//     cublasErrchk(cublasCreate(&cb_handle_));
//     // curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_PHILOX4_32_10)); // DESRES
//     curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
//     gpuErrchk(cudaMalloc((void**)&d_rng_buffer_, coeff_bs.size()*num_dims*sizeof(RealType)));

//     auto end = std::chrono::high_resolution_clock::now();
//     auto seed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

//     curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));

// }


// template<typename RealType> 
// LangevinOptimizer<RealType>::~LangevinOptimizer() {
//     gpuErrchk(cudaFree(d_coeff_bs_));
//     gpuErrchk(cudaFree(d_coeff_cs_));
//     gpuErrchk(cudaFree(d_rng_buffer_));

//     cublasErrchk(cublasDestroy(cb_handle_));
//     curandErrchk(curandDestroyGenerator(cr_rng_));
// }

// template<typename RealType> 
// RealType LangevinOptimizer<RealType>::get_dt() const {
//     return dt_;
// }

// template<typename RealType> 
// void LangevinOptimizer<RealType>::step(
//     const int N,
//     const int D,
//     const int DP,
//     const RealType *dE_dx,
//     const RealType *d2E_dx2, // may be null
//     RealType *d2E_dxdp, // this is modified in place
//     RealType *d_x_t,
//     RealType *d_v_t,
//     RealType *d_dx_dp_t,
//     RealType *d_dv_dp_t,
//     const RealType *d_input_noise_buffer) const {

//     if(N_offset_ == 0) {
//         throw std::runtime_error("bad N_offset");
//     }

//     size_t tpb = 32;
//     size_t n_blocks = (N*D + tpb - 1) / tpb;
//     if(d2E_dx2 != nullptr && d2E_dxdp != nullptr) {

//         hessian_vector_product(N, D, DP, N_offset_, d2E_dx2, d_dx_dp_t, d2E_dxdp);

//         dim3 dimGrid_dxdp(n_blocks, D, DP); // x, y, z dims
//         update_derivatives<RealType><<<dimGrid_dxdp, tpb>>>(
//             coeff_a_,
//             d_coeff_bs_,
//             coeff_d_,
//             d2E_dxdp,
//             dt_,
//             N,
//             D,
//             d_dx_dp_t,
//             d_dv_dp_t
//         );
//         gpuErrchk(cudaPeekAtLastError());
//     }

//     const RealType* d_noise_buf = nullptr;

//     if(d_input_noise_buffer == nullptr) {
//         curandErrchk(templateCurandNormal(cr_rng_, d_rng_buffer_, N*D, 0.0, 1.0));
//         d_noise_buf = d_rng_buffer_;
//     } else {
//         d_noise_buf = d_input_noise_buffer;
//     }

//     dim3 dimGrid_dx(n_blocks, D);
//     update_positions<RealType><<<dimGrid_dx, tpb>>>(
//         d_noise_buf,
//         coeff_a_,
//         d_coeff_bs_,
//         d_coeff_cs_,
//         coeff_d_,
//         coeff_e_,
//         dE_dx,
//         dt_,
//         N,
//         D,
//         d_x_t,
//         d_v_t
//     );

//     gpuErrchk(cudaPeekAtLastError());

// }

// template<typename RealType> 
// void LangevinOptimizer<RealType>::hessian_vector_product(
//     const int N,
//     const int D,
//     const int DP,
//     const int N_offset,
//     const RealType *d_A,
//     // RealType *d_B,
//     // RealType *d_C) const {
//     RealType *d_C,
//     RealType *d_B) const {

//     // sparse routines
//     // convert A/B/C into sparse equivalents
//     int ND = N*D;
//     if(false) {
//         cusparseHandle_t handle = 0;

//         int total_A;
//         cusparseMatDescr_t descr_A = 0;
//         int *d_A_nnz_per_row = 0;
//         cusparseErrchk(cusparseCreate(&handle));
//         gpuErrchk(cudaMalloc((void **)&d_A_nnz_per_row, sizeof(int) * N*D));

//         cusparseErrchk(cusparseCreateMatDescr(&descr_A));
//         cusparseErrchk(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL));
//         cusparseErrchk(cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO));

//         // Initialize Matrices A, B

//         // optimize with lda later if needed
//         cusparseErrchk(cusparseNnz(
//             handle,
//             CUSPARSE_DIRECTION_ROW,
//             N*D,
//             N*D,
//             descr_A,
//             d_A,
//             N*D,
//             d_A_nnz_per_row,
//             &total_A
//         ));

//         int total_B;
//         cusparseMatDescr_t descr_B = 0;
//         int *d_B_nnz_per_row = 0;
//         cusparseErrchk(cusparseCreate(&handle));
//         gpuErrchk(cudaMalloc((void **)&d_B_nnz_per_row, sizeof(int) * N*D));

//         cusparseErrchk(cusparseCreateMatDescr(&descr_B));
//         cusparseErrchk(cusparseSetMatType(descr_B, CUSPARSE_MATRIX_TYPE_GENERAL));
//         cusparseErrchk(cusparseSetMatIndexBase(descr_B, CUSPARSE_INDEX_BASE_ZERO));

//         // optimize with lda later if needed
//         cusparseErrchk(cusparseNnz(
//             handle,
//             CUSPARSE_DIRECTION_ROW,
//             ND,
//             DP,
//             descr_B,
//             d_B,
//             ND,
//             d_B_nnz_per_row,
//             &total_B
//         ));

//         std::cout << "DP: " << DP << " total_B " << total_B << std::endl;

//         RealType *d_sparse_A;
//         int *d_sparse_A_rowptr;
//         int *d_sparse_A_colptr;

//         RealType *d_sparse_B;
//         int *d_sparse_B_rowptr;
//         int *d_sparse_B_colptr;

//         gpuErrchk(cudaMalloc((void **)&d_sparse_A, sizeof(RealType) * total_A));
//         gpuErrchk(cudaMalloc((void **)&d_sparse_A_rowptr, sizeof(int) * (ND + 1)));
//         gpuErrchk(cudaMalloc((void **)&d_sparse_A_colptr, sizeof(int) * total_A));

//         gpuErrchk(cudaMalloc((void **)&d_sparse_B, sizeof(RealType) * total_B));
//         gpuErrchk(cudaMalloc((void **)&d_sparse_B_rowptr, sizeof(int) * (ND + 1)));
//         gpuErrchk(cudaMalloc((void **)&d_sparse_B_colptr, sizeof(int) * total_B));

//         // copy over A and B

//         // cusparseErrchk(cusparseDense2csr(handle, ND, ND, descr_A, d_A, ND, d_A_nnz_per_row,
//             // d_sparse_A, d_sparse_A_rowptr, d_sparse_A_colptr));
 

//         auto start4 = std::chrono::high_resolution_clock::now(); 
//         cusparseErrchk(cusparseDense2csr(handle, N_offset, N_offset, descr_A, d_A, ND, d_A_nnz_per_row,
//             d_sparse_A, d_sparse_A_rowptr, d_sparse_A_colptr));


//         cudaDeviceSynchronize();
//         auto stop4 = std::chrono::high_resolution_clock::now(); 
//         auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - start4); 
//         // std::cout << "dense hessian to sparse duration for hessians: " << duration4.count() << std::endl;

//         auto start3 = std::chrono::high_resolution_clock::now(); 
//         cusparseErrchk(cusparseDense2csr(handle, N_offset, DP, descr_B, d_B, ND, d_B_nnz_per_row,
//             d_sparse_B, d_sparse_B_rowptr, d_sparse_B_colptr));

//         cudaDeviceSynchronize();
//         auto stop3 = std::chrono::high_resolution_clock::now(); 
//         auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3); 
//         // std::cout << "dense hessian to sparse duration for PN3: " << duration3.count() << std::endl;

//         // std::cout << "A" << std::endl;

//         cusparseMatDescr_t descr_C = 0;
//         RealType *d_sparse_C;
//         int *d_sparse_C_rowptr;
//         int *d_sparse_C_colptr;

//         cusparseErrchk(cusparseCreateMatDescr(&descr_C));
//         cusparseErrchk(cusparseSetMatType(descr_C, CUSPARSE_MATRIX_TYPE_GENERAL));
//         cusparseErrchk(cusparseSetMatIndexBase(descr_C, CUSPARSE_INDEX_BASE_ZERO));

//         int baseC, nnzC;
//         // nnzTotalDevHostPtr points to host memory
//         cusparseHandle_t handle2 = 0;
//         cusparseErrchk(cusparseCreate(&handle2));

//         int *nnzTotalDevHostPtr = &nnzC;
//         cusparseSetPointerMode(handle2, CUSPARSE_POINTER_MODE_HOST);
//         gpuErrchk(cudaMalloc((void**)&d_sparse_C_rowptr, sizeof(int)*(ND+1)));
//         // std::cout << "AA " << total_A << " " << total_B <<  std::endl;
//         cusparseErrchk(cusparseXcsrgemmNnz(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, ND, DP, ND,
//                 descr_A, total_A, d_sparse_A_rowptr, d_sparse_A_colptr,
//                 descr_B, total_B, d_sparse_B_rowptr, d_sparse_B_colptr,
//                 descr_C, d_sparse_C_rowptr, nnzTotalDevHostPtr));
//         // std::cout << "B" << std::endl;
//         if (NULL != nnzTotalDevHostPtr){
//             nnzC = *nnzTotalDevHostPtr;
//         } else {
//             gpuErrchk(cudaMemcpy(&nnzC, d_sparse_C_rowptr+ND, sizeof(int), cudaMemcpyDeviceToHost));
//             gpuErrchk(cudaMemcpy(&baseC, d_sparse_C_rowptr, sizeof(int), cudaMemcpyDeviceToHost));
//             nnzC -= baseC;
//         }

//         gpuErrchk(cudaMalloc((void**)&d_sparse_C_colptr, sizeof(int)*nnzC));
//         gpuErrchk(cudaMalloc((void**)&d_sparse_C, sizeof(RealType)*nnzC));

//         auto start = std::chrono::high_resolution_clock::now(); 
//         cusparseErrchk(cusparseCsrgemm(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, ND, DP, ND,
//                 descr_A, total_A, d_sparse_A, d_sparse_A_rowptr, d_sparse_A_colptr,
//                 descr_B, total_B, d_sparse_B, d_sparse_B_rowptr, d_sparse_B_colptr,
//                 descr_C, d_sparse_C, d_sparse_C_rowptr, d_sparse_C_colptr));

//         cudaDeviceSynchronize();
//         auto stop = std::chrono::high_resolution_clock::now(); 
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
//         std::cout << "sparse duration: " << duration.count() << std::endl;
//         throw std::runtime_error("Debug");

//     }

//     RealType alpha = 1.0;
//     RealType beta  = 1.0;   
//     // this is set to UPPER because of fortran ordering
//     // furthermore, we assume the atoms are compacted with ligands at the front, such that we do a subset
//     // where N_offset = (NUM_ATOMS*3)+(NUM_LIGAND_ATOMS)
//     auto start2 = std::chrono::high_resolution_clock::now(); 
//     cublasErrchk(templateSymm(cb_handle_,
//         CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
//         N_offset, DP,
//         &alpha,
//         d_A, ND,
//         d_B, ND,
//         &beta,
//         d_C, ND));

//     cudaDeviceSynchronize();
//     auto stop2 = std::chrono::high_resolution_clock::now();
//     auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2); 
//     // std::cout << "dense duration: " << duration2.count() << std::endl;
// }

// template<typename RealType>
// void LangevinOptimizer<RealType>::set_coeff_a(RealType a) {
//     coeff_a_ = a;
// }

// template<typename RealType>
// void LangevinOptimizer<RealType>::set_coeff_d(RealType d) {
//     coeff_d_ = d;
// }

// template<typename RealType>
// void LangevinOptimizer<RealType>::set_coeff_e(RealType e) {
//     coeff_e_ = e;
// }

// template<typename RealType>
// void LangevinOptimizer<RealType>::set_coeff_b(int num_atoms, const RealType *cb) {
//     gpuErrchk(cudaMemcpy(d_coeff_bs_, cb, num_atoms*sizeof(RealType), cudaMemcpyHostToDevice));
// }

// template<typename RealType>
// void LangevinOptimizer<RealType>::set_coeff_c(int num_atoms, const RealType *cc) {
//     gpuErrchk(cudaMemcpy(d_coeff_cs_, cc, num_atoms*sizeof(RealType), cudaMemcpyHostToDevice));
// }

// template<typename RealType>
// void LangevinOptimizer<RealType>::set_dt(RealType ndt) {
//     dt_ = ndt;
// }

// }

// template class timemachine::LangevinOptimizer<double>;
// template class timemachine::LangevinOptimizer<float>;
