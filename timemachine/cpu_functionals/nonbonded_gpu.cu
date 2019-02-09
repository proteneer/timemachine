#include "nonbonded_gpu.hpp"
#include "constants.hpp"
#include "gpu_utils.cuh"

#include <iostream>

namespace timemachine {

template<typename NumericType>
__global__ void electrostatics_total_derivative(
    const NumericType *coords,
    const NumericType *params, // change to int later?
    const size_t *global_param_idxs, // change to int later?
    const size_t *param_idxs,
    const NumericType *scale_matrix,
    NumericType *energy_out,
    NumericType *grad_out,
    NumericType *hessian_out,
    NumericType *mp_out,
    int P,
    int N) {

    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(i_idx >= N) {
        return;
    }

    NumericType x0 = coords[i_idx*3+0];
    NumericType y0 = coords[i_idx*3+1];
    NumericType z0 = coords[i_idx*3+2];
    NumericType q0 = params[param_idxs[i_idx]];

    // NumericType grad_dx = 0;
    // NumericType grad_dy = 0;
    // NumericType grad_dz = 0;

    NumericType atom_nrg = 0;

    const int n_atoms = N;

    NumericType *mp_out_qi = mp_out + global_param_idxs[param_idxs[i_idx]]*n_atoms*3;

    for(int j_idx=i_idx+1; j_idx < N; j_idx++) {

        NumericType x1 = coords[j_idx*3+0];
        NumericType y1 = coords[j_idx*3+1];
        NumericType z1 = coords[j_idx*3+2];

        NumericType q1 = params[param_idxs[j_idx]];

        NumericType dx = x0 - x1;
        NumericType dy = y0 - y1;
        NumericType dz = z0 - z1;
        NumericType d2x = dx*dx;
        NumericType d2y = dy*dy;
        NumericType d2z = dz*dz;

        NumericType d2ij = d2x + d2y + d2z;
        NumericType dij = sqrt(d2ij);
        NumericType d3ij = dij*dij*dij;
        NumericType d5ij = d3ij*d2ij;

        NumericType sij = scale_matrix[i_idx*N + j_idx];

        atom_nrg += (sij*ONE_4PI_EPS0*q0*q1)/dij;

        NumericType grad_prefactor = (sij*ONE_4PI_EPS0*q0*q1)/d3ij;
        NumericType hess_prefactor = (sij*ONE_4PI_EPS0*q0*q1)/d5ij;


        grad_out[i_idx*3 + 0] += grad_prefactor*(-dx);
        grad_out[i_idx*3 + 1] += grad_prefactor*(-dy);
        grad_out[i_idx*3 + 2] += grad_prefactor*(-dz);

        // optimize to use shared diagonal trick later
        atomicAdd(grad_out + j_idx*3 + 0, grad_prefactor*dx);
        atomicAdd(grad_out + j_idx*3 + 1, grad_prefactor*dy);
        atomicAdd(grad_out + j_idx*3 + 2, grad_prefactor*dz);

        const int x_dim = 0;
        const int y_dim = 1;
        const int z_dim = 2;

        hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim] += hess_prefactor*(-d2ij + 3*d2x);
        hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim] += 3*hess_prefactor*dx*dy;
        hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim] += 3*hess_prefactor*dx*dz;

        hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim] += 3*hess_prefactor*dx*dy;
        hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim] += hess_prefactor*(-d2ij + 3*d2y);
        hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim] += 3*hess_prefactor*dy*dz;

        hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim] += 3*hess_prefactor*dx*dz;
        hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim] += 3*hess_prefactor*dy*dz;
        hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim] += hess_prefactor*(-d2ij + 3*d2z);

        // insanely slow - symmetrize and optimize.
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim, hess_prefactor*(d2ij - 3*d2x));
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim, -3*hess_prefactor*dx*dy);
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim, -3*hess_prefactor*dx*dz);
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim, -3*hess_prefactor*dx*dy);
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim, hess_prefactor*(d2ij - 3*d2y));
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim, -3*hess_prefactor*dy*dz);
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim, -3*hess_prefactor*dx*dz);
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim, -3*hess_prefactor*dy*dz);
        atomicAdd(hessian_out + i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim, hess_prefactor*(d2ij - 3*d2z));

        atomicAdd(hessian_out + j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim, hess_prefactor*(d2ij - 3*d2x));
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim, -3*hess_prefactor*dx*dy);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim, -3*hess_prefactor*dx*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim, hess_prefactor*(-d2ij + 3*d2x));
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim, 3*hess_prefactor*dx*dy);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim, 3*hess_prefactor*dx*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim, -3*hess_prefactor*dx*dy);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim, hess_prefactor*(d2ij - 3*d2y));
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim, -3*hess_prefactor*dy*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim, 3*hess_prefactor*dx*dy);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim, hess_prefactor*(-d2ij + 3*d2y));
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim, 3*hess_prefactor*dy*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim, -3*hess_prefactor*dx*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim, -3*hess_prefactor*dy*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim, hess_prefactor*(d2ij - 3*d2z));
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim, 3*hess_prefactor*dx*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim, 3*hess_prefactor*dy*dz);
        atomicAdd(hessian_out + j_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim, hess_prefactor*(-d2ij + 3*d2z));

        NumericType *mp_out_qj = mp_out + global_param_idxs[param_idxs[j_idx]]*n_atoms*3;

        NumericType PREFACTOR_QI_GRAD = sij*ONE_4PI_EPS0*q1/d3ij;
        NumericType PREFACTOR_QJ_GRAD = sij*ONE_4PI_EPS0*q0/d3ij;

        atomicAdd(mp_out_qi + i_idx*3 + 0, PREFACTOR_QI_GRAD * (-dx));
        atomicAdd(mp_out_qi + i_idx*3 + 1, PREFACTOR_QI_GRAD * (-dy));
        atomicAdd(mp_out_qi + i_idx*3 + 2, PREFACTOR_QI_GRAD * (-dz));
        atomicAdd(mp_out_qi + j_idx*3 + 0, PREFACTOR_QI_GRAD * (dx));
        atomicAdd(mp_out_qi + j_idx*3 + 1, PREFACTOR_QI_GRAD * (dy));
        atomicAdd(mp_out_qi + j_idx*3 + 2, PREFACTOR_QI_GRAD * (dz));

        atomicAdd(mp_out_qj + i_idx*3 + 0, PREFACTOR_QJ_GRAD * (-dx));
        atomicAdd(mp_out_qj + i_idx*3 + 1, PREFACTOR_QJ_GRAD * (-dy));
        atomicAdd(mp_out_qj + i_idx*3 + 2, PREFACTOR_QJ_GRAD * (-dz));
        atomicAdd(mp_out_qj + j_idx*3 + 0, PREFACTOR_QJ_GRAD * (dx));
        atomicAdd(mp_out_qj + j_idx*3 + 1, PREFACTOR_QJ_GRAD * (dy));
        atomicAdd(mp_out_qj + j_idx*3 + 2, PREFACTOR_QJ_GRAD * (dz));

    }

	atomicAdd(energy_out, atom_nrg);

};

template <typename NumericType>
ElectrostaticsGPU<NumericType>::ElectrostaticsGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<NumericType> scale_matrix
) {

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_scale_matrix_, scale_matrix.size()*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &global_param_idxs[0], global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_matrix_, &scale_matrix[0], scale_matrix.size()*sizeof(NumericType), cudaMemcpyHostToDevice));

};


template <typename NumericType>
ElectrostaticsGPU<NumericType>::~ElectrostaticsGPU() {

	gpuErrchk(cudaFree(d_params_));
	gpuErrchk(cudaFree(d_global_param_idxs_));
	gpuErrchk(cudaFree(d_param_idxs_));
	gpuErrchk(cudaFree(d_scale_matrix_));

};


template <typename NumericType>
void ElectrostaticsGPU<NumericType>::total_derivative(
    const size_t n_atoms,
    const size_t n_params,
    const NumericType* d_coords, // [N, 3]
    NumericType* d_energy_out, // []
    NumericType* d_grad_out, // [N,3]
    NumericType* d_hessian_out, // [N, 3, N, 3]
    NumericType* d_mp_out // [P, N, 3]
) {

    size_t tpb = 32;
    size_t n_blocks = (n_atoms + tpb - 1) / tpb;

    electrostatics_total_derivative<<<n_blocks, tpb>>>(
        d_coords,
        d_params_, // change to int later?
        d_global_param_idxs_, // change to int later?
        d_param_idxs_,
        d_scale_matrix_,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out,
        n_params,
        n_atoms);

};


template <typename NumericType>
void ElectrostaticsGPU<NumericType>::total_derivative_cpu(
    const size_t N,
    const size_t P,
    const NumericType* coords, // [N, 3]
    NumericType* energy_out, // []
    NumericType* grad_out, // [N,3]
    NumericType* hessian_out, // [N, 3, N, 3]
    NumericType* mp_out // [P, N, 3]
) {

    NumericType* d_coords; // []
    NumericType* d_energy_out; // []
    NumericType* d_grad_out; // [N,3]
    NumericType* d_hessian_out; // [N, 3, N, 3]
    NumericType* d_mp_out; // [P, N, 3]

    // this is a debugging function.

    gpuErrchk(cudaMalloc((void**)&d_coords, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_energy_out, sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_grad_out, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_hessian_out, N*3*N*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_mp_out, P*N*3*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_coords, coords, N*3*sizeof(NumericType), cudaMemcpyHostToDevice));

    std::cout << "CALLING" << std::endl;

    total_derivative(
    	N,
    	P,
    	d_coords,
    	d_energy_out,
    	d_grad_out,
    	d_hessian_out,
    	d_mp_out);

    gpuErrchk(cudaMemcpy(energy_out, d_energy_out, sizeof(NumericType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(grad_out, d_grad_out, N*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hessian_out, d_hessian_out, N*3*N*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mp_out, d_mp_out, P*N*3*sizeof(NumericType), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_energy_out));
    gpuErrchk(cudaFree(d_grad_out));
    gpuErrchk(cudaFree(d_hessian_out));
    gpuErrchk(cudaFree(d_mp_out));

};

}

template class timemachine::ElectrostaticsGPU<float>;
template class timemachine::ElectrostaticsGPU<double>;
