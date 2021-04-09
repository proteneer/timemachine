#include <vector>
#include <stdexcept>

#include "rmsd_restraint.hpp"
#include "gpu_utils.cuh"
#include <Eigen/Dense>

#include "fixed_point.hpp"
#include "kernels/k_fixed_point.cuh"

#include <chrono>
// using namespace std::chrono;

namespace timemachine {

template<typename RealType>
RMSDRestraint<RealType>::RMSDRestraint(
    const std::vector<int> &atom_map,
    const int N,
    const double k) : h_atom_map_(atom_map), N_(N), k_(k) {
    if(atom_map.size() % 2 != 0) {
        throw std::runtime_error("Bad atom map size!");
    }

    const int B = h_atom_map_.size() / 2;

    gpuErrchk(cudaMalloc(&d_centroid_a_, B*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc(&d_centroid_b_, B*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc(&d_atom_map_, h_atom_map_.size()*sizeof(*d_atom_map_)));
    gpuErrchk(cudaMemcpy(d_atom_map_, &h_atom_map_[0], h_atom_map_.size()*sizeof(int), cudaMemcpyHostToDevice));
}


template<typename RealType>
RMSDRestraint<RealType>::~ RMSDRestraint() {
    gpuErrchk(cudaFree(d_atom_map_));
    gpuErrchk(cudaFree(d_centroid_a_));
    gpuErrchk(cudaFree(d_centroid_b_));
}

template<typename RealType>
void __global__ k_centroid_sums(
    const int * d_atom_map,
    const int B,
    const double * __restrict__ d_x,
    RealType * d_centroid_a,
    RealType * d_centroid_b
    ) {
    const int row_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx >= B || col_idx >= 3) {
        return;
    }
    const int src_idx = d_atom_map[row_idx*2+0];
    atomicAdd(d_centroid_a + row_idx*3 + col_idx, d_x[src_idx*3+col_idx]);

    const int dst_idx = d_atom_map[row_idx*2+1];
    atomicAdd(d_centroid_b + row_idx*3 + col_idx, d_x[dst_idx*3+col_idx]);
}

template<typename RealType>
void __global__ k_finalize(
    const int * d_atom_map,
    const RealType *d_x1,
    const RealType *d_x2,
    const int B,
    const double k,
    const RealType squared_term,
    unsigned long long *d_du_dx_out,
    unsigned long long *d_u_out) {

    const int row_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx >= B || col_idx >= 3) {
        return;
    }

    if(d_u_out && row_idx == 0 && col_idx == 0) {
        atomicAdd(d_u_out, FLOAT_TO_FIXED<RealType>(k*squared_term));
    }

    if(d_du_dx_out) {
        int src_idx = d_atom_map[row_idx*2+0];
        atomicAdd(d_du_dx_out + src_idx*3+col_idx, FLOAT_TO_FIXED<RealType>(k*d_x1[row_idx*3 + col_idx]));

        int dst_idx = d_atom_map[row_idx*2+1];
        atomicAdd(d_du_dx_out + dst_idx*3+col_idx, FLOAT_TO_FIXED<RealType>(k*d_x2[row_idx*3 + col_idx]));
    }
}

template<typename RealType>
void RMSDRestraint<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx, // buffered
    double *d_du_dp,
    unsigned long long *d_du_dl,  // buffered
    unsigned long long *d_u,  // buffered
    cudaStream_t stream) {

    if(N != N_) {
        throw std::runtime_error("N_! = N");
    }
    const int tpb = 32;
    const int B = h_atom_map_.size() / 2;

    dim3 grid_blocks(B, 3);

    gpuErrchk(cudaMemset(d_centroid_a_, 0.0, B*3*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_centroid_b_, 0.0, B*3*sizeof(RealType)));

    k_centroid_sums<RealType><<<grid_blocks, tpb, 0, stream>>>(
        d_atom_map_,
        B,
        d_x,
        d_centroid_a_,
        d_centroid_b_
    );
    gpuErrchk(cudaPeekAtLastError());

    // Need to use RowMajor storage to allow copying
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x1(B, 3);
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x2(B, 3);

    gpuErrchk(cudaMemcpy(x1.data(), d_centroid_a_, B*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(x2.data(), d_centroid_b_, B*3*sizeof(RealType), cudaMemcpyDeviceToHost));

    Eigen::Matrix<RealType, 3, 1> h_a_centroid_mean = x1.colwise().mean();
    Eigen::Matrix<RealType, 3, 1> h_b_centroid_mean = x2.colwise().mean();

    x1 = x1.rowwise() - h_a_centroid_mean.transpose();
    x2 = x2.rowwise() - h_b_centroid_mean.transpose();

    // Keep Matrices in doubles to avoid bad behavior in float32
    Eigen::Matrix<double, 3, 3> c = (x2.transpose().template cast<double> () * x1.template cast<double> ());

    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> svd(c, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto s = svd.singularValues();

    RealType epsilon = 1e-8;

    if(s[0] < epsilon || s[1] < epsilon || s[2] < epsilon) {
        return;
    }

    // Don't think we need these, because they are overridden with a copy at the end
    //gpuErrchk(cudaMemset(d_u_buf_, 0, sizeof(*d_u_buf_)));
    //gpuErrchk(cudaMemset(d_du_dx_buf_, 0, N*3*sizeof(*d_du_dx_buf_)));

    Eigen::Matrix<double, 3, 3> u = svd.matrixU();
    Eigen::Matrix<double, 3, 3> v = svd.matrixV();
    Eigen::Matrix<double, 3, 3> v_t = v.transpose();

    bool is_reflection = u.determinant() * v_t.determinant() < 0.0;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> rotation = u * v_t;

    RealType term = 0;
    if(is_reflection) {
        term = (rotation.trace() + 1)/2 - 1;
    } else {
        term = (rotation.trace() - 1)/2 - 1;
    }

    // backprop'd forces
    Eigen::Matrix<double, 3, 3> eye = Eigen::Matrix<double, 3, 3>::Identity(3, 3);

    RealType squared_term_adjoint = 1.0;
    RealType term_adjoint = squared_term_adjoint*2*term;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> r_a = eye*term_adjoint/2;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u_a = r_a * v_t.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> v_a_t = u.transpose() * r_a;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> smat(3,3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> smat_inv(3,3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> F(3,3);
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            if(i == j) {
                F(i, j) = 0;
                smat(i, j) = s[i];
                smat_inv(i, j) = 1/s[i];
                // s_a_mat(i, j) = s_a
            } else{
                F(i, j) = 1/(s[j]*s[j] - s[i]*s[i]);
                smat(i, j) = 0;
                smat_inv(i, j) = 0;
            }
        }
    }

    // (ytz): taken from here https://j-towns.github.io/papers/svd-derivative.pdf
    // using equations 30-31. note that adjoints of the singular values are always zero,
    // so the middle term is skipped.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> v_a = v_a_t.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lhs_1 = u*F.cwiseProduct(u.transpose()*u_a - u_a.transpose()*u)*smat;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lhs_2 = (eye - u*u.transpose())*u_a*smat_inv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lhs = (lhs_1+lhs_2)*v.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> rhs_1 = smat*(F.cwiseProduct(v.transpose()*v_a - v_a.transpose()*v))*v.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> rhs_2 = smat_inv*v_a.transpose()*(eye - v*v.transpose());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> rhs = u*(rhs_1 + rhs_2);
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> c_adjoint = (lhs + rhs).cast<RealType>();
    // Eigen::RowMajor required to be able to copy out of device
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x2_adjoint = (c_adjoint*x1.transpose()).transpose();
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x1_adjoint = x2*c_adjoint;

    // Reusing d_centroid_a_ and d_centroid_b_ for x1 and x2 adjoints
    gpuErrchk(cudaMemcpy(d_centroid_a_, x1_adjoint.data(), B*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_centroid_b_, x2_adjoint.data(), B*3*sizeof(RealType), cudaMemcpyHostToDevice));

    RealType squared_term = term*term;

    k_finalize<<<grid_blocks, tpb, 0, stream>>>(
        d_atom_map_,
        d_centroid_a_,
        d_centroid_b_,
        B,
        k_,
        squared_term,
        d_du_dx,
        d_u
    );

    gpuErrchk(cudaPeekAtLastError());
};


template class RMSDRestraint<double>;
template class RMSDRestraint<float>;


}
