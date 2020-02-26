#include "stepper.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"

#define PI  3.1415926535897932384626433

#include <iostream>
namespace timemachine {

Stepper::Stepper(int F) : streams_(F) {
    for(int i=0; i < F; i++) {
        gpuErrchk(cudaStreamCreate(&streams_[i]));        
    }

}

Stepper::~Stepper() {
    for(int i=0; i <streams_.size(); i++) {
        gpuErrchk(cudaStreamDestroy(streams_[i]));
    }
}

cudaStream_t Stepper::get_stream(int idx) {
    return streams_[idx];
}

void Stepper::sync_all_streams() {
    for(int i=0; i < streams_.size(); i++) {
        gpuErrchk(cudaStreamSynchronize(streams_[i]));
    }
}

BasicStepper::BasicStepper(
    std::vector<Gradient<3> *> forces
) : count_(0),
    forces_(forces),
    Stepper(forces.size()) {};

void BasicStepper::forward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    unsigned long long *dx) {

    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            coords,
            nullptr,
            params,
            dx, // accumulation place
            nullptr,
            nullptr,
            this->get_stream(f)
        );
    }
    gpuErrchk(cudaDeviceSynchronize());

    count_ += 1;

};

void BasicStepper::backward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    const double *dx_tangent,
    double *coords_jvp,
    double *params_jvp) {

    count_ -= 1;


    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            coords,
            dx_tangent,
            params,
            nullptr, 
            coords_jvp,
            params_jvp,
            this->get_stream(f)
        );
    }
    gpuErrchk(cudaDeviceSynchronize());


};



LambdaStepper::LambdaStepper(
    std::vector<Gradient <4> *> forces,
    const std::vector<double> &lambda_schedule,
    const std::vector<int> &lambda_flags,
    const int exponent
) : forces_(forces),
    lambda_schedule_(lambda_schedule),
    exponent_(exponent),
    count_(0),
    Stepper(forces.size()) {

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


__global__ void convert_3d_to_4d(
    const int N,
    const double *d_coords_3d,
    const int *lambda_flags, // [1, 0, or -1]
    const double lambda,
    const int k,
    const double *d_coords_3d_tangent, // can be nullptr
    const double du_dl_adjoint, // used only if d_coords_3d_tangent is not null
    double *d_coords_4d,
    double *d_coords_4d_tangent // must be nullptr if tangent is nullptr
) { 

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx_3d = atom_idx*3 + d_idx;
    int local_idx_4d = atom_idx*4 + d_idx;

    if(d_idx == 3) {

        double w;
        if(lambda_flags[atom_idx] == 1) {
            w = tan(lambda*(PI/2))/k;
        } else if (lambda_flags[atom_idx] == -1) {
            w = tan(-(lambda-1)*(PI/2))/k;
        } else {
            w = 0;
        }
        d_coords_4d[local_idx_4d] = w;

        if(d_coords_3d_tangent) {
            double dw;
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

__global__ void accumulate_dU_dl(
    const int N,
    const unsigned long long *d_forces_4d,
    const int *lambda_flags, // [1, 0, or -1]
    const double lambda,
    const int k,
    double *du_dl_buffer) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    if(d_idx == 3) {

        int local_idx_4d = atom_idx*4 + d_idx;

        double dw;
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

        double du_dw = static_cast<double>(static_cast<long long>(d_forces_4d[local_idx_4d]))/FIXED_EXPONENT;

        atomicAdd(du_dl_buffer, dw*du_dw);

    }
}

void LambdaStepper::forward_step_host(
    const int N,
    const int P,
    const double *h_coords, // 3d
    const double *h_params, // 3d
    unsigned long long *h_dx) {

    double *d_coords, *d_params;
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

void LambdaStepper::forward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    unsigned long long *dx) {

    const int D = 4;

    size_t tpb = 32;
    size_t n_blocks = (N*D + tpb - 1) / tpb;
    dim3 dimGrid(n_blocks, D);

    if(count_ > lambda_schedule_.size() - 1) {
        throw std::runtime_error("backward step bad counter!");
    }

    convert_3d_to_4d<<<dimGrid, tpb>>>(
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

    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            d_coords_buffer_,
            nullptr,
            params,
            d_forces_buffer_, // accumulation place
            nullptr,
            nullptr,
            this->get_stream(f)
        );
    }
    gpuErrchk(cudaDeviceSynchronize());

    accumulate_dU_dl<<<dimGrid, tpb>>>(N, d_forces_buffer_, d_lambda_flags_, lambda_schedule_[count_], exponent_, &d_du_dl_[count_]);
    gpuErrchk(cudaPeekAtLastError());
    convert_4d_to_3d<<<dimGrid, tpb>>>(N, d_forces_buffer_, dx);
    gpuErrchk(cudaPeekAtLastError());
    count_ += 1;

};

void LambdaStepper::get_du_dl(
    double *buf) {
    const int T = get_T();
    cudaMemcpy(buf, d_du_dl_, T*sizeof(double), cudaMemcpyDeviceToHost);
};


void LambdaStepper::set_du_dl_adjoint(
    const int T,
    const double *adj) {
    if(T != lambda_schedule_.size()) {
        throw std::runtime_error("adjoint size not the same as lambda schedule size");
    }
    du_dl_adjoint_.resize(T);
    memcpy(&du_dl_adjoint_[0], adj, T*sizeof(double));
};

void LambdaStepper::backward_step(
    const int N,
    const int P,
    const double *coords,
    const double *params,
    const double *dx_tangent,
    double *coords_jvp,
    double *params_jvp) {
    // first decrement
    count_ -= 1;

    const int D = 4;

    size_t tpb = 32;
    size_t n_blocks = (N*D + tpb - 1) / tpb;
    dim3 dimGrid(n_blocks, D);

    if(count_ > lambda_schedule_.size() - 1) {
        throw std::runtime_error("backward step bad counter!");
    }

    convert_3d_to_4d<<<dimGrid, tpb>>>(
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


    gpuErrchk(cudaDeviceSynchronize());
    for(int f=0; f < forces_.size(); f++) {
        forces_[f]->execute_device(
            N,
            P,
            d_coords_buffer_,
            d_dx_tangent_buffer_,
            params,
            nullptr,
            d_coords_jvp_buffer_,
            params_jvp,
            this->get_stream(f)
        );
    }
    gpuErrchk(cudaDeviceSynchronize());

    convert_4d_to_3d<<<dimGrid, tpb>>>(N, d_coords_jvp_buffer_, coords_jvp);

    gpuErrchk(cudaPeekAtLastError());



};

void LambdaStepper::backward_step_host(
    const int N,
    const int P,
    const double *h_coords, // 3d
    const double *h_params,
    const double *h_dx_tangent,
    double *h_coords_jvp,
    double *h_params_jvp) {

    double *d_coords, *d_params, *d_dx_tangent, *d_coords_jvp, *d_params_jvp;

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

// template class LambdaStepper<double>;
// template class LambdaStepper<float>;

// template class BasicStepper<double>;
// template class BasicStepper<float>;

};
