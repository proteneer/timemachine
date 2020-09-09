#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
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
    nblist_(lambda_plane_idxs.size(), 3) {

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

    // gpuErrchk(cudaMalloc(&d_lambda_group_idxs_, N_*sizeof(*d_lambda_group_idxs_)));
    // gpuErrchk(cudaMemcpy(d_lambda_group_idxs_, &lambda_group_idxs[0], N_*sizeof(*d_lambda_group_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lj_scales_, E_*sizeof(*d_lj_scales_)));
    gpuErrchk(cudaMemcpy(d_lj_scales_, &lj_scales[0], E_*sizeof(*d_lj_scales_), cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMalloc(&d_lj_params_, N_*2*sizeof(*d_lj_params_)));
    // gpuErrchk(cudaMemcpy(d_lj_params_, &lj_params[0], N_*2*sizeof(*d_lj_params_), cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMalloc(&d_du_dlj_primals_, N_*2*sizeof(*d_du_dlj_primals_)));
    // gpuErrchk(cudaMemset(d_du_dlj_primals_, 0, N_*2*sizeof(*d_du_dlj_primals_)));

    // gpuErrchk(cudaMalloc(&d_du_dlj_tangents_, N_*2*sizeof(*d_du_dlj_tangents_)));
    // gpuErrchk(cudaMemset(d_du_dlj_tangents_, 0, N_*2*sizeof(*d_du_dlj_tangents_)));


};

template <typename RealType>
LennardJones<RealType>::~LennardJones() {

    // gpuErrchk(cudaFree(d_lj_params_));
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_lj_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

    // gpuErrchk(cudaFree(d_lambda_group_idxs_));
    // gpuErrchk(cudaFree(d_du_dlj_primals_));
    // gpuErrchk(cudaFree(d_du_dlj_tangents_));
};


// template <typename RealType>
// void LennardJones<RealType>::get_du_dlj_primals(double *buf) {
//     gpuErrchk(cudaMemcpy(buf, d_du_dlj_primals_, N_*2*sizeof(*d_du_dlj_primals_), cudaMemcpyDeviceToHost));
// }

// template <typename RealType>
// void LennardJones<RealType>::get_du_dlj_tangents(double *buf) {
//     gpuErrchk(cudaMemcpy(buf, d_du_dlj_tangents_, N_*2*sizeof(*d_du_dlj_tangents_), cudaMemcpyDeviceToHost));
// }

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

    // its safe for us to build a neighborlist in a lower dimension.
    nblist_.compute_block_bounds(N_, D, d_x, stream);

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();

    // these can be ran in two streams later on
    k_lennard_jones_inference<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_x,
        d_p,
        d_box,
        lambda,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        cutoff_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u
    );

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


// template <typename RealType>
// void LennardJones<RealType>::execute_lambda_jvp_device(
//     const int N,
//     const double *d_coords_primals,
//     const double *d_coords_tangents,
//     const double lambda_primal,
//     const double lambda_tangent,
//     double *d_out_coords_primals,
//     double *d_out_coords_tangents,
//     cudaStream_t stream) {

//     if(N != N_) {
//         throw std::runtime_error("N != N_");
//     }

//     const int tpb = 32;
//     const int B = (N_+tpb-1)/tpb;
//     const int D = 3;

//     nblist_.compute_block_bounds(N_, D, d_coords_primals, stream);

//     gpuErrchk(cudaPeekAtLastError());

//     dim3 dimGrid(B, B, 1); // x, y, z dims
//     dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

//     auto start = std::chrono::high_resolution_clock::now();

//     k_lennard_jones_jvp<RealType><<<dimGrid, tpb, 0, stream>>>(
//         N_,
//         d_coords_primals,
//         d_coords_tangents,
//         lambda_primal,
//         lambda_tangent,
//         d_lambda_plane_idxs_,
//         d_lambda_offset_idxs_,
//         d_lambda_group_idxs_,
//         d_lj_params_,
//         cutoff_,
//         nblist_.get_block_bounds_ctr(),
//         nblist_.get_block_bounds_ext(),
//         d_out_coords_primals,
//         d_out_coords_tangents,
//         d_du_dlj_primals_,
//         d_du_dlj_tangents_
//     );

//     // cudaDeviceSynchronize();
//     gpuErrchk(cudaPeekAtLastError());

//     if(E_ > 0) {
//         k_lennard_jones_exclusion_jvp<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
//             E_,
//             d_coords_primals,
//             d_coords_tangents,
//             lambda_primal,
//             lambda_tangent,
//             d_lambda_plane_idxs_,
//             d_lambda_offset_idxs_,
//             d_lambda_group_idxs_,
//             d_exclusion_idxs_,
//             d_lj_scales_,
//             d_lj_params_,
//             cutoff_,
//             d_out_coords_primals,
//             d_out_coords_tangents,
//             d_du_dlj_primals_,
//             d_du_dlj_tangents_
//         );            

//         // cudaDeviceSynchronize();
//         gpuErrchk(cudaPeekAtLastError());
//     }



// };

template class LennardJones<double>;
template class LennardJones<float>;

} // namespace timemachine