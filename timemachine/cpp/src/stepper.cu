#include "stepper.hpp"
#include "kernel_utils.cuh"
#include "fixed_point.hpp"

#define PI 3.14159265358979323846

#include <iostream>
namespace timemachine {

template<typename RealType>
BasicStepper<RealType>::BasicStepper(
    std::vector<Gradient <RealType, 3> *> forces
) : count_(0),
    forces_(forces) {};

template<typename RealType>
void BasicStepper<RealType>::forward_step(
    const int N,
    const int P,
    const RealType *coords,
    const RealType *params,
    unsigned long long *dx) {

    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            coords,
            nullptr,
            params,
            dx, // accumulation place
            nullptr,
            nullptr);
    }

    count_ += 1;

};

template<typename RealType>
void BasicStepper<RealType>::backward_step(
    const int N,
    const int P,
    const RealType *coords,
    const RealType *params,
    const RealType *dx_tangent,
    RealType *coords_jvp,
    RealType *params_jvp) {

    count_ -= 1;

    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            coords,
            dx_tangent,
            params,
            nullptr, 
            coords_jvp,
            params_jvp
        );
    }


};

template class BasicStepper<double>;


// template<typename RealType>
// __global__ void compute_dw_dl(
//     const int *lambda_flags, // [1, 0, or -1]
//     const RealType lambda,
//     RealType *dw_dl) {

//     if(d_idx == 3) {
//         RealType upper = pow(lambda, exponent);
//         RealType lower = (1-upper);
//         RealType w;
//         if(lambda_flags[atom_idx] == 1) {
//             w = upper/lower;
//         } else if (lambda_flags[atom_idx] == -1) {
//             w = lower/upper;
//         } else {
//             w = 0;
//         }
//         d_coords_4d[local_idx_4d] = w;
//     } else {
//         d_coords_4d[local_idx_4d] = d_coords_3d[local_idx_3d];
//     }

// };

template<typename RealType>
LambdaStepper<RealType>::LambdaStepper(
    std::vector<Gradient <RealType, 4> *> forces,
    const std::vector<RealType> &lambda_schedule,
    const std::vector<int> &lambda_flags,
    const int exponent
) : forces_(forces),
    lambda_schedule_(lambda_schedule),
    exponent_(exponent),
    count_(0) {

    const int N = lambda_flags.size();
    const int D = 4;

    gpuErrchk(cudaMalloc(&d_coords_buffer_, N*D*sizeof(*d_coords_buffer_)));
    gpuErrchk(cudaMalloc(&d_dx_tangent_buffer_, N*D*sizeof(*d_dx_tangent_buffer_)));
    gpuErrchk(cudaMalloc(&d_coords_jvp_buffer_, N*D*sizeof(*d_coords_jvp_buffer_)));
    gpuErrchk(cudaMalloc(&d_forces_buffer_, N*D*sizeof(*d_forces_buffer_)));
    gpuErrchk(cudaMalloc(&d_lambda_flags_, N*sizeof(*d_lambda_flags_)));
    gpuErrchk(cudaMemcpy(d_lambda_flags_, &lambda_flags[0], N*sizeof(*d_lambda_flags_), cudaMemcpyHostToDevice));

    const int T = lambda_schedule_.size();

    gpuErrchk(cudaMalloc(&d_du_dl_, T*sizeof(*d_du_dl_)));
    gpuErrchk(cudaMemset(d_du_dl_, 0, T*sizeof(*d_du_dl_)));
}


// also call to setup tangents
template<typename RealType>
__global__ void convert_3d_to_4d(
    const int N,
    const RealType *d_coords_3d,
    const int *lambda_flags, // [1, 0, or -1]
    const RealType lambda,
    const int k,
    const RealType *d_coords_3d_tangent, // can be nullptr
    const RealType du_dl_adjoint, // used only if d_coords_3d_tangent is not null
    RealType *d_coords_4d,
    RealType *d_coords_4d_tangent // must be nullptr if tangent is nullptr
) { 

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx_3d = atom_idx*3 + d_idx;
    int local_idx_4d = atom_idx*4 + d_idx;

    if(d_idx == 3) {

        RealType w;
        if(lambda_flags[atom_idx] == 1) {
            w = tan(lambda*(PI/2))/k;
        } else if (lambda_flags[atom_idx] == -1) {
            w = tan(-(lambda-1)*(PI/2))/k;
        } else {
            w = 0;
        }
        d_coords_4d[local_idx_4d] = w;

        if(d_coords_3d_tangent) {
            RealType dw;
            if(lambda_flags[atom_idx] == 1) {
                auto cosw = cos(lambda*PI/2);
                auto secw = 1/cosw;
                dw = (secw*secw*PI)/(2*k);
            } else if (lambda_flags[atom_idx] == -1) { 
                auto sinw = sin(lambda*PI/2);
                auto cscw = 1/sinw;
                dw = -(cscw*cscw*PI)/(2*k);
            } else {
                dw = 0;
            }
            d_coords_4d_tangent[local_idx_4d] = dw*du_dl_adjoint;
        }
    } else {
        d_coords_4d[local_idx_4d] = d_coords_3d[local_idx_3d];
        if(d_coords_3d_tangent) {
            d_coords_4d_tangent[local_idx_4d] = d_coords_3d_tangent[local_idx_3d];
        }
    }

}

template<typename T>
__global__ void convert_4d_to_3d(
    const int N,
    const T *d_forces_4d,
    T *d_forces_3d) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    if(d_idx < 3) {
        int local_idx_3d = atom_idx*3 + d_idx;
        int local_idx_4d = atom_idx*4 + d_idx;
        d_forces_3d[local_idx_3d] = d_forces_4d[local_idx_4d];
    }

}

template <typename RealType>
__global__ void accumulate_dU_dl(
    const int N,
    const unsigned long long *d_forces_4d,
    const int *lambda_flags, // [1, 0, or -1]
    const RealType lambda,
    const int k,
    RealType *du_dl_buffer) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    if(d_idx == 3) {

        int local_idx_4d = atom_idx*4 + d_idx;

        RealType dw;
        if(lambda_flags[atom_idx] == 1) {
            auto cosw = cos(lambda*PI/2);
            auto secw = 1/cosw;
            dw = (secw*secw*PI)/(2*k);
        } else if (lambda_flags[atom_idx] == -1) { 
            auto sinw = sin(lambda*PI/2);
            auto cscw = 1/sinw;
            dw = -(cscw*cscw*PI)/(2*k);
        } else {
            dw = 0;
        }

        RealType du_dw = static_cast<RealType>(static_cast<long long>(d_forces_4d[local_idx_4d]))/FIXED_EXPONENT;

        atomicAdd(du_dl_buffer, dw*du_dw);

    }
}

template<typename RealType>
void LambdaStepper<RealType>::forward_step_host(
    const int N,
    const int P,
    const RealType *h_coords, // 3d
    const RealType *h_params, // 3d
    unsigned long long *h_dx) {

    RealType *d_coords, *d_params;
    unsigned long long *d_dx;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_coords, D*N*sizeof(*d_coords)));
    gpuErrchk(cudaMalloc(&d_params, P*sizeof(*d_params)));
    gpuErrchk(cudaMalloc(&d_dx, D*N*sizeof(*d_dx)));

    gpuErrchk(cudaMemcpy(d_coords, h_coords, D*N*sizeof(*d_coords), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, h_params, P*sizeof(*d_params), cudaMemcpyHostToDevice));

    this->forward_step(
        N,
        P,
        d_coords,
        d_params,
        d_dx
    );

    gpuErrchk(cudaMemcpy(h_dx, d_dx, N*D*sizeof(*d_dx), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_params));
    gpuErrchk(cudaFree(d_dx));

};

template<typename RealType>
void LambdaStepper<RealType>::forward_step(
    const int N,
    const int P,
    const RealType *coords,
    const RealType *params,
    unsigned long long *dx) {

    const int D = 4;

    size_t tpb = 32;
    size_t n_blocks = (N*D + tpb - 1) / tpb;
    dim3 dimGrid(n_blocks, D);

    if(count_ > lambda_schedule_.size() - 1) {
        throw std::runtime_error("backward step bad counter!");
    }

    convert_3d_to_4d<RealType><<<dimGrid, tpb>>>(
        N,
        coords,
        d_lambda_flags_,
        lambda_schedule_[count_],
        exponent_,
        nullptr,
        0,
        d_coords_buffer_,
        nullptr
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemset(d_forces_buffer_, 0, N*D*sizeof(*d_forces_buffer_)));

    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            d_coords_buffer_,
            nullptr,
            params,
            d_forces_buffer_, // accumulation place
            nullptr,
            nullptr);
    }

    accumulate_dU_dl<<<dimGrid, tpb>>>(N, d_forces_buffer_, d_lambda_flags_, lambda_schedule_[count_], exponent_, &d_du_dl_[count_]);
    gpuErrchk(cudaPeekAtLastError());
    convert_4d_to_3d<<<dimGrid, tpb>>>(N, d_forces_buffer_, dx);
    gpuErrchk(cudaPeekAtLastError());
    count_ += 1;

};

template<typename RealType>
void LambdaStepper<RealType>::get_du_dl(
    RealType *buf) {
    const int T = get_T();
    cudaMemcpy(buf, d_du_dl_, T*sizeof(RealType), cudaMemcpyDeviceToHost);
};


template<typename RealType>
void LambdaStepper<RealType>::set_du_dl_adjoint(
    const int T,
    const RealType *adj) {
    if(T != lambda_schedule_.size()) {
        throw std::runtime_error("adjoint size not the same as lambda schedule size");
    }
    du_dl_adjoint_.resize(T);
    memcpy(&du_dl_adjoint_[0], adj, T*sizeof(RealType));
};

template<typename RealType>
void LambdaStepper<RealType>::backward_step(
    const int N,
    const int P,
    const RealType *coords,
    const RealType *params,
    const RealType *dx_tangent,
    RealType *coords_jvp,
    RealType *params_jvp) {
    // first decrement
    count_ -= 1;

    const int D = 4;

    size_t tpb = 32;
    size_t n_blocks = (N*D + tpb - 1) / tpb;
    dim3 dimGrid(n_blocks, D);

    if(count_ > lambda_schedule_.size() - 1) {
        throw std::runtime_error("backward step bad counter!");
    }

    convert_3d_to_4d<RealType><<<dimGrid, tpb>>>(
        N,
        coords,
        d_lambda_flags_,
        lambda_schedule_[count_],
        exponent_,
        dx_tangent,
        du_dl_adjoint_[count_],
        d_coords_buffer_,
        d_dx_tangent_buffer_
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemset(d_coords_jvp_buffer_, 0, N*D*sizeof(*d_coords_jvp_buffer_)));
    gpuErrchk(cudaMemset(params_jvp, 0, P*sizeof(*params_jvp)));

    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            d_coords_buffer_,
            d_dx_tangent_buffer_,
            params,
            nullptr,
            d_coords_jvp_buffer_,
            params_jvp // no need to reshape this
        );
    }

    convert_4d_to_3d<<<dimGrid, tpb>>>(N, d_coords_jvp_buffer_, coords_jvp);

    gpuErrchk(cudaPeekAtLastError());



};

template<typename RealType>
void LambdaStepper<RealType>::backward_step_host(
    const int N,
    const int P,
    const RealType *h_coords, // 3d
    const RealType *h_params,
    const RealType *h_dx_tangent,
    RealType *h_coords_jvp,
    RealType *h_params_jvp) {

    RealType *d_coords, *d_params, *d_dx_tangent, *d_coords_jvp, *d_params_jvp;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_coords, D*N*sizeof(*d_coords)));
    gpuErrchk(cudaMalloc(&d_params, P*sizeof(*d_params)));
    gpuErrchk(cudaMalloc(&d_dx_tangent, D*N*sizeof(*d_dx_tangent)));

    gpuErrchk(cudaMemcpy(d_coords, h_coords, D*N*sizeof(*d_coords), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dx_tangent, h_dx_tangent, D*N*sizeof(*d_dx_tangent), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, h_params, P*sizeof(*d_params), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_coords_jvp, D*N*sizeof(*d_coords)));
    gpuErrchk(cudaMalloc(&d_params_jvp, P*sizeof(*d_params)));

    this->backward_step(
        N,
        P,
        d_coords,
        d_params,
        d_dx_tangent,
        d_coords_jvp,
        d_params_jvp
    );

    gpuErrchk(cudaMemcpy(h_coords_jvp, d_coords_jvp, N*D*sizeof(*d_coords_jvp), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_params_jvp, d_params_jvp, P*sizeof(*d_params_jvp), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_params));
    gpuErrchk(cudaFree(d_dx_tangent));
    gpuErrchk(cudaFree(d_coords_jvp));
    gpuErrchk(cudaFree(d_params_jvp));

};

template class LambdaStepper<double>;


};