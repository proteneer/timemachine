#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>

#include "lennard_jones.hpp"
#include "gpu_utils.cuh"

#include "k_lennard_jones.cuh"
#include "k_lennard_jones_jvp.cuh"


namespace timemachine {

template <typename RealType>
LennardJones<RealType>::LennardJones(
    // const std::vector<double> &lj_params, // [N,2]
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &lj_scales, // [E]
    const std::vector<int> &lambda_plane_idxs, // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    // const std::vector<int> &lambda_group_idxs, // [N]
    double cutoff
) :  N_(lambda_plane_idxs.size()),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_plane_idxs.size()),
    d_perm_(nullptr) {

    if(lambda_plane_idxs.size() != N_) {
        throw std::runtime_error("lambda plane idxs need to have size N");
    }

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(lj_scales.size()*2 != exclusion_idxs.size()) {
        throw std::runtime_error("charge scale idxs size not half of exclusion size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_*sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_*sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lj_scales_, E_*sizeof(*d_lj_scales_)));
    gpuErrchk(cudaMemcpy(d_lj_scales_, &lj_scales[0], E_*sizeof(*d_lj_scales_), cudaMemcpyHostToDevice));

};

template <typename RealType>
LennardJones<RealType>::~LennardJones() {
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_lj_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
    gpuErrchk(cudaFree(d_perm_));
};


// stackoverflow is your best friend
// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
std::vector<int> argsort(const std::vector<T> &v) {
  std::vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});
  return idx;
}


template <typename RealType>
void LennardJones<RealType>::execute_device(
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

    if(N != N_) {
        std::ostringstream err_msg;
        err_msg << "N != N_ " << N << " " << N_;
        throw std::runtime_error(err_msg.str());
    }

    const int tpb = 32;
    const int B = (N_+tpb-1)/tpb;
    const int D = 3;

    // sort the particles once based on epsilons
    if(d_perm_ == nullptr) {

        gpuErrchk(cudaMalloc(&d_perm_, N_*sizeof(*d_perm_)));

        std::vector<double> lj_params(N*2);
        gpuErrchk(cudaMemcpy(&lj_params[0], d_p, N*2*sizeof(*d_p), cudaMemcpyDeviceToHost));
        std::vector<double> eps_params(N);
        for(int i=0; i < N; i++) {
            eps_params[i] = lj_params[2*i+1];
        }

        std::vector<int> perm = argsort(eps_params);
        gpuErrchk(cudaMemcpy(d_perm_, &perm[0], N*sizeof(*d_perm_), cudaMemcpyHostToDevice));
    }

    // is this correct?
    // nblist_.compute_block_bounds(
    //     N_,
    //     D,
    //     d_x,
    //     d_box,
    //     nullptr,
    //     stream
    // );

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();

    // these can be ran in two streams later on

    // int *run_counter;
    // gpuErrchk(cudaMallocManaged(&run_counter, sizeof(int)))

    k_lennard_jones_inference<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_x,
        d_p,
        d_box,
        lambda,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        cutoff_,
        // nblist_.get_block_bounds_ctr(),
        // nblist_.get_block_bounds_ext(),
        nullptr,
        nullptr,
        d_perm_,
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u
        // run_counter
    );

    cudaDeviceSynchronize();

    // std::cout << "RUN COUNTER: " << *run_counter << std::endl;

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if(E_ > 0) {
        k_lennard_jones_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_x,
            d_p,
            d_box,
            lambda,
            d_lambda_plane_idxs_,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_lj_scales_,
            cutoff_,
            d_du_dx,
            d_du_dp,
            d_du_dl,
            d_u
        );
        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
}

template class LennardJones<double>;
template class LennardJones<float>;

} // namespace timemachine
