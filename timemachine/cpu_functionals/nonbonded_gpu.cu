#include "nonbonded_gpu.hpp"
#include "electrostatics.cuh"
#include "lennard_jones.cuh"
#include "gpu_utils.cuh"


#include <ctime>
#include <iostream>


namespace timemachine {



template <typename NumericType>
ElectrostaticsGPU<NumericType>::ElectrostaticsGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<NumericType> scale_matrix
) : P_(params.size()), N_(param_idxs.size()) {

    // convert to int version
    std::vector<int> int_global_param_idxs;
    for(auto a : global_param_idxs) {
        int_global_param_idxs.push_back(a);
    }
    std::vector<int> int_param_idxs;
    for(auto a : param_idxs) {
        int_param_idxs.push_back(a);
    }

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, int_global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, int_param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_scale_matrix_, scale_matrix.size()*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &int_global_param_idxs[0], int_global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &int_param_idxs[0], int_param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
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
        n_atoms);

};

template <typename NumericType>
void ElectrostaticsGPU<NumericType>::set_params(const std::vector<NumericType> &params) {
    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
}

template <typename NumericType>
std::vector<NumericType> ElectrostaticsGPU<NumericType>::get_params() const {
    std::vector<NumericType> buf(P_);
    gpuErrchk(cudaMemcpy(&buf[0], d_params_, P_*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return buf;
}

template <typename NumericType>
std::vector<int> ElectrostaticsGPU<NumericType>::get_param_idxs() const {
    std::vector<int> buf(N_);
    gpuErrchk(cudaMemcpy(&buf[0], d_param_idxs_, N_*sizeof(int), cudaMemcpyDeviceToHost));
    return buf;
}

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

    gpuErrchk(cudaMemset(d_energy_out, 0, sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_grad_out, 0, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_hessian_out, 0, N*3*N*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_mp_out, 0, P*N*3*sizeof(NumericType)));

    cudaDeviceSynchronize();
    std::clock_t start; double duration; start = std::clock();

    total_derivative(
        N,
        P,
        d_coords,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out);

    cudaDeviceSynchronize();

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"ES_DURATION: "<< duration <<'\n';

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

template <typename NumericType>
LennardJonesGPU<NumericType>::LennardJonesGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<NumericType> scale_matrix
) : P_(params.size()) {

    // convert to int version
    std::vector<int> int_global_param_idxs;
    for(auto a : global_param_idxs) {
        int_global_param_idxs.push_back(a);
    }
    std::vector<int> int_param_idxs;
    for(auto a : param_idxs) {
        int_param_idxs.push_back(a);
    }

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, int_global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, int_param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_scale_matrix_, scale_matrix.size()*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &int_global_param_idxs[0], int_global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &int_param_idxs[0], int_param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_matrix_, &scale_matrix[0], scale_matrix.size()*sizeof(NumericType), cudaMemcpyHostToDevice));

};

template <typename NumericType>
LennardJonesGPU<NumericType>::~LennardJonesGPU() {

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_global_param_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_scale_matrix_));

};

template <typename NumericType>
void LennardJonesGPU<NumericType>::set_params(const std::vector<NumericType> &params) {
    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
}


// refactor into interface class
template <typename NumericType>
void LennardJonesGPU<NumericType>::total_derivative(
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

    lennard_jones_total_derivative<<<n_blocks, tpb>>>(
        d_coords,
        d_params_, // change to int later?
        d_global_param_idxs_, // change to int later?
        d_param_idxs_,
        d_scale_matrix_,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out,
        n_atoms);

};

template <typename NumericType>
std::vector<NumericType> LennardJonesGPU<NumericType>::get_params() const {
    std::vector<NumericType> buf(P_);
    gpuErrchk(cudaMemcpy(&buf[0], d_params_, P_*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return buf;
}

template <typename NumericType>
void LennardJonesGPU<NumericType>::total_derivative_cpu(
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

    gpuErrchk(cudaMemset(d_energy_out, 0, sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_grad_out, 0, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_hessian_out, 0, N*3*N*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_mp_out, 0, P*N*3*sizeof(NumericType)));

    cudaDeviceSynchronize();
    std::clock_t start; double duration; start = std::clock();


    total_derivative(
        N,
        P,
        d_coords,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out);

    cudaDeviceSynchronize();

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"LJ_DURATION: "<< duration <<'\n';

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

template class timemachine::LennardJonesGPU<float>;
template class timemachine::LennardJonesGPU<double>;
