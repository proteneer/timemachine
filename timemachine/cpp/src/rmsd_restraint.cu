#include <vector>
#include <stdexcept>

#include "rmsd_restraint.hpp"
#include "gpu_utils.cuh"
#include <Eigen/Dense>

#include "fixed_point.hpp"
// #include "k_fixed_point.hpp"

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

    gpuErrchk(cudaMalloc(&d_u_buf_, sizeof(d_u_buf_)));
    gpuErrchk(cudaMalloc(&d_du_dx_buf_, N*3*sizeof(d_du_dx_buf_)));

}


template<typename RealType>
RMSDRestraint<RealType>::~ RMSDRestraint() {
    gpuErrchk(cudaFree(d_u_buf_));
    gpuErrchk(cudaFree(d_du_dx_buf_));
}

void __global__ k_finalize(
    const int N,
    const unsigned long long *d_du_dx_buf,
    const unsigned long long *d_u_buf,
    unsigned long long *d_du_dx_out,
    unsigned long long *d_u_out) {

    const int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(d_u_out && atom_idx == 0) {
        atomicAdd(d_u_out, *d_u_buf);
    }


    if(atom_idx >= N) {
        return;
    }

    if(d_du_dx_out) {
        for(int d=0; d < 3; d++) {
            atomicAdd(d_du_dx_out + atom_idx*3 + d, d_du_dx_buf[atom_idx*3 + d]);
        }
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

    std::vector<double> h_x(N*3);

    gpuErrchk(cudaMemcpy(&h_x[0], d_x, h_x.size()*sizeof(*d_x), cudaMemcpyDeviceToHost));

    const int B = h_atom_map_.size() / 2;

    Eigen::MatrixXd x1(B,3);
    Eigen::MatrixXd x2(B,3);

    // compute centroids etc. on the GPU directly

    for(int b=0; b < B; b++) {

        int src_idx = h_atom_map_[b*2+0];
        x1(b, 0) = h_x[src_idx*3+0];
        x1(b, 1) = h_x[src_idx*3+1];
        x1(b, 2) = h_x[src_idx*3+2];

        int dst_idx = h_atom_map_[b*2+1];
        x2(b, 0) = h_x[dst_idx*3+0];
        x2(b, 1) = h_x[dst_idx*3+1];
        x2(b, 2) = h_x[dst_idx*3+2];
    }


    Eigen::Vector3d h_a_centroid = x1.colwise().mean();
    Eigen::Vector3d h_b_centroid = x2.colwise().mean();

    x1 = x1.rowwise() - h_a_centroid.transpose();
    x2 = x2.rowwise() - h_b_centroid.transpose();

    Eigen::MatrixXd c = x2.transpose() * x1;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(c, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto s = svd.singularValues();

    double epsilon = 1e-8;

    gpuErrchk(cudaMemset(d_u_buf_, 0, sizeof(*d_u_buf_)));
    gpuErrchk(cudaMemset(d_du_dx_buf_, 0, N*3*sizeof(*d_du_dx_buf_)));

    if(s[0] < epsilon || s[1] < epsilon || s[2] < epsilon) {
        return;
    }

    Eigen::MatrixXd u = svd.matrixU();
    Eigen::MatrixXd v = svd.matrixV();
    Eigen::MatrixXd v_t = v.transpose();

    bool is_reflection = u.determinant() * v_t.determinant() < 0.0;

    Eigen::MatrixXd rotation = u * v_t;

    double term = 0;
    if(is_reflection) {
        return;
    } else {
        term = (rotation.trace() - 1)/2 - 1;
    }

    double squared_term = term*term;

    unsigned long long nrg = static_cast<unsigned long long>(llrint(k_*squared_term*FIXED_EXPONENT));

    // backprop'd forces
    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(3, 3);

    double squared_term_adjoint = 1.0;
    double term_adjoint = squared_term_adjoint*2*term;
    Eigen::MatrixXd r_a = eye*term_adjoint/2;
    Eigen::MatrixXd u_a = r_a * v_t.transpose();
    Eigen::MatrixXd v_a_t = u.transpose() * r_a;

    Eigen::MatrixXd smat(3,3);
    Eigen::MatrixXd smat_inv(3,3);
    Eigen::MatrixXd F(3,3);
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
    Eigen::MatrixXd v_a = v_a_t.transpose();
    Eigen::MatrixXd lhs_1 = u*F.cwiseProduct(u.transpose()*u_a - u_a.transpose()*u)*smat;
    Eigen::MatrixXd lhs_2 = (eye - u*u.transpose())*u_a*smat_inv;
    Eigen::MatrixXd lhs = (lhs_1+lhs_2)*v.transpose();
    Eigen::MatrixXd rhs_1 = smat*(F.cwiseProduct(v.transpose()*v_a - v_a.transpose()*v))*v.transpose();
    Eigen::MatrixXd rhs_2 = smat_inv*v_a.transpose()*(eye - v*v.transpose());
    Eigen::MatrixXd rhs = u*(rhs_1 + rhs_2);
    Eigen::MatrixXd c_adjoint = lhs + rhs;
    Eigen::MatrixXd x2_adjoint = (c_adjoint*x1.transpose()).transpose();
    Eigen::MatrixXd x1_adjoint = x2*c_adjoint;

    std::vector<unsigned long long> du_dx(N*3, 0);

    for(int b=0; b < B; b++) {
        int src_idx = h_atom_map_[b*2+0];
        du_dx[src_idx*3+0] += static_cast<unsigned long long>(llrint(k_*x1_adjoint(b, 0)*FIXED_EXPONENT));
        du_dx[src_idx*3+1] += static_cast<unsigned long long>(llrint(k_*x1_adjoint(b, 1)*FIXED_EXPONENT));
        du_dx[src_idx*3+2] += static_cast<unsigned long long>(llrint(k_*x1_adjoint(b, 2)*FIXED_EXPONENT));

        int dst_idx = h_atom_map_[b*2+1];
        du_dx[dst_idx*3+0] += static_cast<unsigned long long>(llrint(k_*x2_adjoint(b, 0)*FIXED_EXPONENT));
        du_dx[dst_idx*3+1] += static_cast<unsigned long long>(llrint(k_*x2_adjoint(b, 1)*FIXED_EXPONENT));
        du_dx[dst_idx*3+2] += static_cast<unsigned long long>(llrint(k_*x2_adjoint(b, 2)*FIXED_EXPONENT));
    }

    gpuErrchk(cudaMemcpy(d_u_buf_, &nrg, sizeof(*d_u_buf_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_du_dx_buf_, &du_dx[0], N*3*sizeof(*d_du_dx_buf_), cudaMemcpyHostToDevice));

    const int blocks= (N_+32-1)/32;
    const int tpb = 32;

    k_finalize<<<blocks, tpb, 0, stream>>>(
        N,
        d_du_dx_buf_,
        d_u_buf_,
        d_du_dx,
        d_u
    );

    gpuErrchk(cudaPeekAtLastError());

};


template class RMSDRestraint<double>;
template class RMSDRestraint<float>;


}