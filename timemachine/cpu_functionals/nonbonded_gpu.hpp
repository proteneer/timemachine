#pragma once

#include <vector>

namespace timemachine {


template <typename NumericType>
class ElectrostaticsGPU {

private:

    NumericType* d_params_;
    size_t* d_global_param_idxs_;
    size_t* d_param_idxs_;
    NumericType* d_scale_matrix_;

public:

    ElectrostaticsGPU(
        std::vector<NumericType> params,
        std::vector<size_t> global_param_idxs,
        std::vector<size_t> param_idxs,
        std::vector<NumericType> scale_matrix
    );

    ~ElectrostaticsGPU();

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
