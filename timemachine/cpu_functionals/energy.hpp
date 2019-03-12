#pragma once

#include<vector>

namespace timemachine {

template <typename NumericType>
class EnergyGPU {

public:

    virtual void total_derivative(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* d_coords, // [N, 3]
        NumericType* d_energy_out, // []
        NumericType* d_grad_out, // [N,3]
        NumericType* d_hessian_out, // [N, 3, N, 3]
        NumericType* d_mp_out // [P, N, 3]
    ) = 0;

    virtual void total_derivative_cpu(
        const size_t n_atoms,
        const size_t n_params,
        const NumericType* d_coords, // [N, 3]
        NumericType* d_energy_out, // []
        NumericType* d_grad_out, // [N,3]
        NumericType* d_hessian_out, // [N, 3, N, 3]
        NumericType* d_mp_out // [P, N, 3]
    ) = 0;

    virtual void set_params(
        const std::vector<NumericType> &new_params
    ) = 0;


    virtual std::vector<NumericType> get_params() const = 0;

};

}