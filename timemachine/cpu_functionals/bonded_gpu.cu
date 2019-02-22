#include "bonded_gpu.hpp"
// #include "electrostatics.cuh"
// #include "lennard_jones.cuh"
#include "harmonic_bond.cuh"
#include "harmonic_angle.cuh"
#include "periodic_torsion.cuh"
#include "gpu_utils.cuh"


#include <ctime>
#include <iostream>


namespace timemachine {

template <typename NumericType>
HarmonicBondGPU<NumericType>::HarmonicBondGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<size_t> bond_idxs
) : n_bonds_(bond_idxs.size()/2) {

    // convert to int version
    std::vector<int> int_global_param_idxs;
    for(auto a : global_param_idxs) {
        int_global_param_idxs.push_back(a);
    }
    std::vector<int> int_param_idxs;
    for(auto a : param_idxs) {
        int_param_idxs.push_back(a);
    }
    std::vector<int> int_bond_idxs;
    for(auto a : bond_idxs) {
        int_bond_idxs.push_back(a);
    }

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, int_global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, int_param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_bond_idxs_, bond_idxs.size()*sizeof(*d_bond_idxs_)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &int_global_param_idxs[0], int_global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &int_param_idxs[0], int_param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &int_bond_idxs[0], int_bond_idxs.size()*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

};


template <typename NumericType>
HarmonicBondGPU<NumericType>::~HarmonicBondGPU() {

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_global_param_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_bond_idxs_));

};


template <typename NumericType>
void HarmonicBondGPU<NumericType>::total_derivative(
    const size_t n_atoms,
    const size_t n_params,
    const NumericType* d_coords, // [N, 3]
    NumericType* d_energy_out, // []
    NumericType* d_grad_out, // [N,3]
    NumericType* d_hessian_out, // [N, 3, N, 3]
    NumericType* d_mp_out // [P, N, 3]
) {

    size_t tpb = 32;
    size_t n_blocks = (n_bonds_ + tpb - 1) / tpb;

    harmonic_bond_total_derivative<<<n_blocks, tpb>>>(
        d_coords,
        d_params_, // change to int later?
        d_global_param_idxs_, // change to int later?
        d_param_idxs_,
        d_bond_idxs_,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out,
        n_atoms,
        n_bonds_
    );

};


template <typename NumericType>
void HarmonicBondGPU<NumericType>::total_derivative_cpu(
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

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"BOND_DURATION: "<< duration <<'\n';

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
HarmonicAngleGPU<NumericType>::HarmonicAngleGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<size_t> angle_idxs
) : n_angles_(angle_idxs.size()/3) {

    // convert to int version
    std::vector<int> int_global_param_idxs;
    for(auto a : global_param_idxs) {
        int_global_param_idxs.push_back(a);
    }
    std::vector<int> int_param_idxs;
    for(auto a : param_idxs) {
        int_param_idxs.push_back(a);
    }
    std::vector<int> int_angle_idxs;
    for(auto a : angle_idxs) {
        int_angle_idxs.push_back(a);
    }

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, int_global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, int_param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_angle_idxs_, angle_idxs.size()*sizeof(*d_angle_idxs_)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &int_global_param_idxs[0], int_global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &int_param_idxs[0], int_param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_angle_idxs_, &int_angle_idxs[0], int_angle_idxs.size()*sizeof(*d_angle_idxs_), cudaMemcpyHostToDevice));

};


template <typename NumericType>
HarmonicAngleGPU<NumericType>::~HarmonicAngleGPU() {

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_global_param_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_angle_idxs_));

};


template <typename NumericType>
void HarmonicAngleGPU<NumericType>::total_derivative(
    const size_t n_atoms,
    const size_t n_params,
    const NumericType* d_coords, // [N, 3]
    NumericType* d_energy_out, // []
    NumericType* d_grad_out, // [N,3]
    NumericType* d_hessian_out, // [N, 3, N, 3]
    NumericType* d_mp_out // [P, N, 3]
) {

    size_t tpb = 32;
    size_t n_blocks = (n_angles_ + tpb - 1) / tpb;

    harmonic_angle_total_derivative<<<n_blocks, tpb>>>(
        d_coords,
        d_params_, // change to int later?
        d_global_param_idxs_, // change to int later?
        d_param_idxs_,
        d_angle_idxs_,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out,
        n_atoms,
        n_angles_
    );

};


template <typename NumericType>
void HarmonicAngleGPU<NumericType>::total_derivative_cpu(
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

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"ANGLE_DURATION: "<< duration <<'\n';

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
PeriodicTorsionGPU<NumericType>::PeriodicTorsionGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<size_t> torsion_idxs
) : n_torsions_(torsion_idxs.size()/4) {

    // convert to int version
    std::vector<int> int_global_param_idxs;
    for(auto a : global_param_idxs) {
        int_global_param_idxs.push_back(a);
    }
    std::vector<int> int_param_idxs;
    for(auto a : param_idxs) {
        int_param_idxs.push_back(a);
    }
    std::vector<int> int_torsion_idxs;
    for(auto a : torsion_idxs) {
        int_torsion_idxs.push_back(a);
    }

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, int_global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, int_param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_torsion_idxs_, torsion_idxs.size()*sizeof(*d_torsion_idxs_)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &int_global_param_idxs[0], int_global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &int_param_idxs[0], int_param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_torsion_idxs_, &int_torsion_idxs[0], int_torsion_idxs.size()*sizeof(*d_torsion_idxs_), cudaMemcpyHostToDevice));

};


template <typename NumericType>
PeriodicTorsionGPU<NumericType>::~PeriodicTorsionGPU() {

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_global_param_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_torsion_idxs_));

};


template <typename NumericType>
void PeriodicTorsionGPU<NumericType>::total_derivative(
    const size_t n_atoms,
    const size_t n_params,
    const NumericType* d_coords, // [N, 3]
    NumericType* d_energy_out, // []
    NumericType* d_grad_out, // [N,3]
    NumericType* d_hessian_out, // [N, 3, N, 3]
    NumericType* d_mp_out // [P, N, 3]
) {

    size_t tpb = 32;
    size_t n_blocks = (n_torsions_ + tpb - 1) / tpb;

    periodic_torsion_total_derivative<<<n_blocks, tpb>>>(
        d_coords,
        d_params_, // change to int later?
        d_global_param_idxs_, // change to int later?
        d_param_idxs_,
        d_torsion_idxs_,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out,
        n_atoms,
        n_torsions_
    );

};


template <typename NumericType>
void PeriodicTorsionGPU<NumericType>::total_derivative_cpu(
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

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"PERIODIC TORSION: "<< duration <<'\n';

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



template class timemachine::HarmonicBondGPU<float>;
template class timemachine::HarmonicBondGPU<double>;

template class timemachine::HarmonicAngleGPU<float>;
template class timemachine::HarmonicAngleGPU<double>;

template class timemachine::PeriodicTorsionGPU<float>;
template class timemachine::PeriodicTorsionGPU<double>;