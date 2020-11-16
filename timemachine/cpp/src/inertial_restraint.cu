#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include "inertial_restraint.hpp"
#include "gpu_utils.cuh"
#include "solver.hpp"
#include "../fixed_point.hpp"

namespace timemachine {

template <typename RealType>
InertialRestraint<RealType>::InertialRestraint(
    const std::vector<int> &group_a_idxs,
    const std::vector<int> &group_b_idxs,
    const std::vector<double> &masses,
    const double k) : N_(masses.size()),
    N_A_(group_a_idxs.size()),
    N_B_(group_b_idxs.size()),
    k_(k),
    h_a_idxs_(group_a_idxs),
    h_b_idxs_(group_b_idxs),
    h_masses_(masses),
    h_x_buffer_(N_*3),
    h_conf_adjoint_(N_*3) {

    for(int i=0; i < group_a_idxs.size(); i++) {
        if(group_a_idxs[i] >= N_ || group_a_idxs[i] < 0) {
            throw std::runtime_error("Invalid group_a_idx!");
        }
        h_c_idxs_.push_back(group_a_idxs[i]);
    }

    for(int i=0; i < group_b_idxs.size(); i++) {
        if(group_b_idxs[i] >= N_ || group_b_idxs[i] < 0) {
            throw std::runtime_error("Invalid group_a_idx!");
        }
        h_c_idxs_.push_back(group_b_idxs[i]);
    }

    // (ytz): take care of special corner case when a_idxs and b_idxs
    // are not disjoint
    std::set<int> c_set(h_c_idxs_.begin(), h_c_idxs_.end());
    h_c_idxs_.clear();
    for(auto idx : c_set) {
        h_c_idxs_.push_back(idx);
    }

    N_C_ = h_c_idxs_.size();

    gpuErrchk(cudaMalloc(&d_c_idxs_, N_C_*sizeof(*d_c_idxs_)));
    gpuErrchk(cudaMemcpy(d_c_idxs_, &h_c_idxs_[0], N_C_*sizeof(*d_c_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMallocHost(&h_x_memcpy_buf_pinned_, N_C_*3*sizeof(*h_x_memcpy_buf_pinned_)));
    gpuErrchk(cudaMalloc(&d_x_memcpy_buf_, N_C_*3*sizeof(*d_x_memcpy_buf_)));

};

template <typename RealType>
InertialRestraint<RealType>::~InertialRestraint() {
    gpuErrchk(cudaFree(d_c_idxs_));
    gpuErrchk(cudaFree(d_x_memcpy_buf_));
    gpuErrchk(cudaFreeHost(h_x_memcpy_buf_pinned_));
};

// center of mass inertia tensor
void inertia_tensor(
    const int NX,
    const int *h_idxs,
    const double *h_masses,
    const double *h_x_in,
    double *out_tensor) {

    double centroid[3] = {0};
    double sum = 0;

    for(int i=0; i < NX; i++) {
        int atom_idx = h_idxs[i];
        double mass = h_masses[atom_idx];
        for(int d=0; d < 3; d++) {
            centroid[d] += mass*h_x_in[atom_idx*3+d];
        }
        sum += mass;
    }

    centroid[0] /= sum;
    centroid[1] /= sum;
    centroid[2] /= sum;

    double xx = 0;
    double xy = 0;
    double xz = 0;
    double yy = 0;
    double yz = 0;
    double zz = 0;

    for(int i=0; i < NX; i++) {
        int atom_idx = h_idxs[i];
        double mass = h_masses[atom_idx];
        double ci[3];
        for(int d=0; d < 3; d++) {
            ci[d] = h_x_in[atom_idx*3+d] - centroid[d];
        }
        xx += mass * (ci[1] * ci[1] + ci[2] * ci[2]);
        yy += mass * (ci[0] * ci[0] + ci[2] * ci[2]);
        zz += mass * (ci[1] * ci[1] + ci[0] * ci[0]);
        xy -= mass * ci[0] * ci[1];
        xz -= mass * ci[0] * ci[2];
        yz -= mass * ci[2] * ci[1];
    }

    out_tensor[0*3+0] = xx/sum;
    out_tensor[0*3+1] = xy/sum;
    out_tensor[0*3+2] = xz/sum;
    out_tensor[1*3+0] = xy/sum;
    out_tensor[1*3+1] = yy/sum;
    out_tensor[1*3+2] = yz/sum;
    out_tensor[2*3+0] = xz/sum;
    out_tensor[2*3+1] = yz/sum;
    out_tensor[2*3+2] = zz/sum;


}

__global__ void k_atomic_add(double *addr, double var) {
    atomicAdd(addr, var);
}

__global__ void k_gather_x(
    const double *src,
    const int C,
    const int *c_idxs,
    double *dst) {

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid >= C) {
        return;
    }

    const int dim = blockIdx.y;

    dst[tid*3+dim] = src[c_idxs[tid]*3+dim];

}

__global__ void k_accumulate_scatter(
    const int C,
    const int *c_idxs,
    const double *src,
    unsigned long long *dst) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid >= C) {
        return;
    }
    const int dim = blockIdx.y;
    atomicAdd(dst + c_idxs[tid]*3 + dim, static_cast<unsigned long long>((long long) (src[tid*3 + dim]*FIXED_EXPONENT)));

}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


void print_matrix(double x[3][3], std::string name) {

    std::cout << "matrix: " << name << std::endl;
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            std::cout << x[i][j] << " ";
        }
        std::cout << std::endl;
    }

}

void grad_inertia_tensor(
    int N,
    const int *idxs,
    const double *masses,
    const double *conf,
    const double adjoint_tensor[3][3],
    double *conf_adjoint) {

    double centroid[3] = {0};
    double sum = 0;

    for(int i=0; i < N; i++) {
        int atom_idx = idxs[i];
        double mass = masses[atom_idx];
        for(int d=0; d < 3; d++) {
            centroid[d] += mass*conf[atom_idx*3+d];
        }
        sum += mass;
    }

    centroid[0] /= sum;
    centroid[1] /= sum;
    centroid[2] /= sum;

    double dxx = adjoint_tensor[0][0];
    double dxy = adjoint_tensor[0][1];
    double dxz = adjoint_tensor[0][2];
    double dyy = adjoint_tensor[1][1];
    double dyz = adjoint_tensor[1][2];
    double dzz = adjoint_tensor[2][2];

    double mass_sum = 0;
    for(int i=0; i < N; i++) {
        mass_sum += masses[idxs[i]];
    }

    for(int i=0; i < N; i++) {
        int a_idx = idxs[i];
        double mass = masses[a_idx];
        double xs = conf[a_idx*3+0] - centroid[0];
        double ys = conf[a_idx*3+1] - centroid[1];
        double zs = conf[a_idx*3+2] - centroid[2];
        conf_adjoint[a_idx*3+0] += (dyy*2*xs + dzz*2*xs + -dxy*2*ys + -dxz*2*zs)*(mass/mass_sum);
        conf_adjoint[a_idx*3+1] += (dzz*2*ys + dxx*2*ys + -dxy*2*xs + -dyz*2*zs)*(mass/mass_sum);
        conf_adjoint[a_idx*3+2] += (dxx*2*zs + dyy*2*zs + -dxz*2*xs + -dyz*2*ys)*(mass/mass_sum);
    }


}

void grad_eigh(
    const double w[3],       // eigenvalues
    const double v[3][3],    // eigenvectors
    const double vg[3][3],   // eigenvector adjoints
    double a_adjoint[3][3]   // input array adjoints
    ) {
    /*
    (ytz): I really hate this code. See timemachine.lib.pmi.grad_eigh for a slightly more
    readable python implementation.

    Reference implementation of the vector jacobian product of the derivative of column
    eigenvectors with respect to the input matrix. This code is derived from perturbation theory,
    and originally ported over from autograd in a long series of conversions. 
    */

    double off_diag[3][3] = {
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
    };

    double F[3][3];

    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            F[i][j] = off_diag[i][j] / (w[j] - w[i] + (1-off_diag[i][j]));
        }
    }

    double C[3][3] = {0};

    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            double sum = 0;
            for(int k=0; k < 3; k++) {
                sum += v[k][i]*vg[k][j];
            }
            C[i][j] = sum*F[i][j];
        }
    }


    double D[3][3] = {0};
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            double sum = 0;
            for(int k=0; k < 3; k++) {
                sum += v[i][k]*C[k][j];
            }
            D[i][j] = sum;
        }
    }

    double vjp_temp[3][3] = {0};

    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            double sum = 0;
            for(int k=0; k < 3; k++) {
                sum += D[i][k]*v[j][k];
            }
            vjp_temp[i][j] = sum;
        }
    }

    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            if(i == j) {
                a_adjoint[i][j] = vjp_temp[i][j];
            } else {
                a_adjoint[i][j] = (vjp_temp[i][j] + vjp_temp[j][i])/2;
            }
        }
    }

}

template <typename RealType>
void InertialRestraint<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    double *d_du_dl,
    double *d_u,
    cudaStream_t stream) {

    int tpb = 32;

    // (ytz): This function proceeds as follows:
    // 0. (GPU) Gather a subset of the coordinates that will be used
    // 1. (GPU->CPU) Copy only coordinates for atoms in c_idxs
    // 2. (CPU) Scatter gathered coordinates 
    // 3. (CPU) Compute the 3x3 inertia tensor (real and symmetric) for each set of indices..
    // 4. (CPU) Solve for the eigenvalues and eigenvector analytically. Sorted in ascending order.
    // 5. (CPU) Compute the energy function function using the paired column eigenvectors.
    // 6. (CPU) Backpropagate the derivative into a single [Nx3] array.
    // 7. (CPU) Gather the forces
    // 8. (CPU->GPU) Copy only the forces for atoms in c_idxs
    // 9. (GPU) Update forces and energy

    // cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();

    dim3 dimGather((N_C_+tpb-1)/tpb, 3, 1);

    k_gather_x<<<dimGather, tpb>>>(d_x, N_C_, d_c_idxs_, d_x_memcpy_buf_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_x_memcpy_buf_pinned_, d_x_memcpy_buf_, N_C_*3*sizeof(*d_x_memcpy_buf_), cudaMemcpyDeviceToHost));    

    std::vector<double> &h_x_in = h_x_buffer_;
    for(int i=0; i < h_c_idxs_.size(); i++) {
        for(int d=0; d < 3; d++) {
            h_x_in[h_c_idxs_[i]*3+d] = h_x_memcpy_buf_pinned_[i*3+d];
        }
    }

    const std::vector<double> &h_masses = h_masses_;
    const std::vector<int> &h_a_idxs = h_a_idxs_;
    const std::vector<int> &h_b_idxs = h_b_idxs_;

    std::vector<double> a_tensor(3*3);
    std::vector<double> b_tensor(3*3);

    inertia_tensor(N_A_, &h_a_idxs[0], &h_masses[0], &h_x_in[0], &a_tensor[0]);
    inertia_tensor(N_B_, &h_b_idxs[0], &h_masses[0], &h_x_in[0], &b_tensor[0]);

    double (&a_array)[3][3] = *reinterpret_cast<double (*)[3][3]>(&a_tensor[0]);
    double a_w[3]; // eigenvalues
    double a_v[3][3]; // eigenvectors

    dsyevv3(a_array, a_v, a_w);

    double (&b_array)[3][3] = *reinterpret_cast<double (*)[3][3]>(&b_tensor[0]);
    double b_w[3]; // eigenvalues
    double b_v[3][3]; // eigenvectors

    dsyevv3(b_array, b_v, b_w);

    // this is equivalent to:
    // R' = matmul(A^T, B)
    // sum_i (1 - dot(R'[i], e[i]))^2 where e is the identity matrix (the standard basis)
    // see reference python code for more information
    double loss = 0;

    double dl_da_v[3][3]; // derivatives of loss wrt. a's eigenvectors
    double dl_db_v[3][3]; // derivatives of loss wrt. b's eigenvectors

    for(int j=0; j < 3; j++) {
        double dot_prod = 0;
        for(int i=0; i < 3; i++) {
            dot_prod += a_v[i][j] * b_v[i][j];
        }

        double delta = 1 - abs(dot_prod);
        loss += delta*delta;
        double prefactor = -sgn(dot_prod)*2*delta*k_;
        for(int i=0; i < 3; i++) {
            dl_da_v[i][j] = prefactor*b_v[i][j];
            dl_db_v[i][j] = prefactor*a_v[i][j];
        }
    }

    double dl_da_tensor[3][3];
    double dl_db_tensor[3][3];

    grad_eigh(a_w, a_v, dl_da_v, dl_da_tensor);
    grad_eigh(b_w, b_v, dl_db_v, dl_db_tensor);

    for(int i=0; i < h_c_idxs_.size(); i++) {
        for(int d=0; d < 3; d++) {
            h_conf_adjoint_[h_c_idxs_[i]*3+d] = 0;
        }
    }

    grad_inertia_tensor(N_A_, &h_a_idxs[0], &h_masses[0], &h_x_in[0], dl_da_tensor, &h_conf_adjoint_[0]);
    grad_inertia_tensor(N_B_, &h_b_idxs[0], &h_masses[0], &h_x_in[0], dl_db_tensor, &h_conf_adjoint_[0]);

    for(int i=0; i < h_c_idxs_.size(); i++) {
        for(int d=0; d < 3; d++) {
            h_x_memcpy_buf_pinned_[i*3+d] = h_conf_adjoint_[h_c_idxs_[i]*3+d];
        }
    }

    gpuErrchk(cudaMemcpy(d_x_memcpy_buf_, h_x_memcpy_buf_pinned_, N_C_*3*sizeof(*d_x_memcpy_buf_), cudaMemcpyHostToDevice));

    if(d_u) {
        k_atomic_add<<<1, 1, 0>>>(d_u, loss*k_);        
        gpuErrchk(cudaPeekAtLastError());
    }

    const int B = (N+tpb-1)/tpb;
    dim3 dimGrid(B, 3, 1);

    gpuErrchk(cudaPeekAtLastError());

    if(d_du_dx) {
        k_accumulate_scatter<<<dimGrid, tpb, 0>>>(
            N_C_,
            d_c_idxs_,
            d_x_memcpy_buf_,
            d_du_dx
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "total: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;;
    gpuErrchk(cudaPeekAtLastError());

};

template class InertialRestraint<double>;
template class InertialRestraint<float>;

} // namespace timemachine