#pragma once

#include <vector>
#include "energy.hpp"

namespace timemachine {


template <typename NumericType>
class ElectrostaticsGPU: public EnergyGPU<NumericType> {

private:

    NumericType* d_params_;
    int* d_global_param_idxs_;
    int* d_param_idxs_;
    NumericType* d_scale_matrix_;
    int P_;
    int N_;

public:

    ElectrostaticsGPU(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<NumericType> scale_matrix
    );

    ~ElectrostaticsGPU();

    void set_params(const std::vector<NumericType> &new_params);

    std::vector<NumericType> get_params() const;

    std::vector<int> get_param_idxs() const;

    void total_derivative(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* d_coords, // [N, 3]
        NumericType* d_energy_out, // []
        NumericType* d_grad_out, // [N,3]
        NumericType* d_hessian_out, // [N, 3, N, 3]
        NumericType* d_mp_out // [P, N, 3]
    );

    void total_derivative_cpu(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* d_coords, // [N, 3]
        NumericType* d_energy_out, // []
        NumericType* d_grad_out, // [N,3]
        NumericType* d_hessian_out, // [N, 3, N, 3]
        NumericType* d_mp_out // [P, N, 3]
    );

};



template <typename NumericType>
class LennardJonesGPU: public EnergyGPU<NumericType> {

private:

    NumericType* d_params_;
    int* d_global_param_idxs_;
    int* d_param_idxs_;
    NumericType* d_scale_matrix_;
    int P_;

public:

    LennardJonesGPU(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<NumericType> scale_matrix
    );

    ~LennardJonesGPU();

    void set_params(const std::vector<NumericType> &new_params);

    std::vector<NumericType> get_params() const;

    void total_derivative(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* d_coords, // [N, 3]
        NumericType* d_energy_out, // []
        NumericType* d_grad_out, // [N,3]
        NumericType* d_hessian_out, // [N, 3, N, 3]
        NumericType* d_mp_out // [P, N, 3]
    );

    void total_derivative_cpu(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* d_coords, // [N, 3]
        NumericType* d_energy_out, // []
        NumericType* d_grad_out, // [N,3]
        NumericType* d_hessian_out, // [N, 3, N, 3]
        NumericType* d_mp_out // [P, N, 3]
    );

};


}
