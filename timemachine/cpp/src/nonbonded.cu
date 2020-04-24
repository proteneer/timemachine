#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "nonbonded.hpp"
#include "gpu_utils.cuh"

#include "k_nonbonded_deterministic_lambda.cuh"
// #include "k_nonbonded_deterministic_lambda.cuh"

namespace timemachine {

template <typename RealType, int D>
Nonbonded<RealType, D>::Nonbonded(
    const std::vector<int> &charge_param_idxs,
    const std::vector<int> &lj_param_idxs,
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<int> &charge_scale_idxs, // [E]
    const std::vector<int> &lj_scale_idxs, // [E]
    const std::vector<int> &lambda_idxs, // [N]
    double cutoff
) :  N_(charge_param_idxs.size()),
    cutoff_(cutoff),
    E_(charge_scale_idxs.size()),
    nblist_(charge_param_idxs.size(), D) {

    if(lambda_idxs.size() != N_) {
        throw std::runtime_error("lambda idxs need to have size N");
    }

    if(charge_scale_idxs.size()*2 != exclusion_idxs.size()) {
        throw std::runtime_error("charge scale idxs size not half of exclusion size!");
    }

    if(charge_scale_idxs.size() != lj_scale_idxs.size()) {
        throw std::runtime_error("Charge scale idxs does not match LJ scale idxs!");
    }

    if(charge_param_idxs.size()*2 != lj_param_idxs.size()) {
        throw std::runtime_error("Charge param idxs not half of lj param idxs!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_idxs_, N_*sizeof(*d_lambda_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_idxs_, &lambda_idxs[0], N_*sizeof(*d_lambda_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_scale_idxs_, E_*sizeof(*d_charge_scale_idxs_)));
    gpuErrchk(cudaMemcpy(d_charge_scale_idxs_, &charge_scale_idxs[0], E_*sizeof(*d_charge_scale_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lj_scale_idxs_, E_*sizeof(*d_lj_scale_idxs_)));
    gpuErrchk(cudaMemcpy(d_lj_scale_idxs_, &lj_scale_idxs[0], E_*sizeof(*d_lj_scale_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_param_idxs_, N_*sizeof(*d_charge_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_charge_param_idxs_, &charge_param_idxs[0], N_*sizeof(*d_charge_param_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lj_param_idxs_, N_*2*sizeof(*d_lj_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_lj_param_idxs_, &lj_param_idxs[0], N_*2*sizeof(*d_lj_param_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_tmp_coords_4d_, N_*4*sizeof(d_tmp_coords_4d_)));
    gpuErrchk(cudaMalloc(&d_tmp_grads_4d_, N_*4*sizeof(d_tmp_grads_4d_)));

};

template <typename RealType, int D>
Nonbonded<RealType, D>::~Nonbonded() {
    gpuErrchk(cudaFree(d_charge_param_idxs_));
    gpuErrchk(cudaFree(d_lj_param_idxs_));
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_charge_scale_idxs_));
    gpuErrchk(cudaFree(d_lj_scale_idxs_));
    gpuErrchk(cudaFree(d_lambda_idxs_));
    gpuErrchk(cudaFree(d_tmp_coords_4d_));
    gpuErrchk(cudaFree(d_tmp_grads_4d_));
};


template <typename RealType, int D>
void Nonbonded<RealType, D>::execute_lambda_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_coords_tangents,
    const double *d_params,
    const double lambda,
    const double lambda_tangent,
    unsigned long long *d_out_coords,
    double *d_out_du_dl,
    double *d_out_coords_tangents,
    double *d_out_params_tangents,
    cudaStream_t stream
) {

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    int tpb = 32;
    int B = (N_+tpb-1)/tpb;

    // its safe for us to build a neighborlist in a lower dimension.
    nblist_.compute_block_bounds(N_, D, d_coords, stream);

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();
    if(d_coords_tangents == nullptr) {
        
        gpuErrchk(cudaMemset(d_tmp_grads_4d_, 0, N_*4*sizeof(d_tmp_grads_4d_)));

        // these can be ran in two streams later on
        k_nonbonded_inference<RealType><<<dimGrid, tpb, 0, stream>>>(
            N_,
            d_coords,
            d_params,
            lambda,
            d_lambda_idxs_,
            d_charge_param_idxs_,
            d_lj_param_idxs_,
            cutoff_,
            nblist_.get_block_bounds_ctr(),
            nblist_.get_block_bounds_ext(),
            d_out_coords,
            d_out_du_dl
        );

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        if(E_ > 0) {
            k_nonbonded_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
                E_,
                d_coords,
                d_params,
                lambda,
                d_lambda_idxs_,
                d_exclusion_idxs_,
                d_charge_scale_idxs_,
                d_lj_scale_idxs_,
                d_charge_param_idxs_,
                d_lj_param_idxs_,
                cutoff_,
                d_out_coords,
                d_out_du_dl
            );
            // cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
        }

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "Nonbonded Elapsed time: " << elapsed.count() << " s\n";

    } else {

        // // do *not* accumulate tangents here
        k_nonbonded_jvp<RealType><<<dimGrid, tpb, 0, stream>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            lambda,
            lambda_tangent,
            d_lambda_idxs_,
            d_charge_param_idxs_,
            d_lj_param_idxs_,
            cutoff_,
            nblist_.get_block_bounds_ctr(),
            nblist_.get_block_bounds_ext(),
            d_out_coords_tangents,
            d_out_params_tangents
        );

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        if(E_ > 0) {
            k_nonbonded_exclusion_jvp<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
                E_,
                d_coords,
                d_coords_tangents,
                d_params,
                lambda,
                lambda_tangent,
                d_lambda_idxs_,
                d_exclusion_idxs_,
                d_charge_scale_idxs_,
                d_lj_scale_idxs_,
                d_charge_param_idxs_,
                d_lj_param_idxs_,
                cutoff_,
                d_out_coords_tangents,
                d_out_params_tangents
            );            

            // cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
        }


        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "Nonbonded JVP Elapsed time: " << elapsed.count() << " s\n";

    }


};


template <typename RealType, int D>
void Nonbonded<RealType, D>::execute_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_coords_tangents,
    const double *d_params,
    unsigned long long *d_out_coords,
    double *d_out_coords_tangents,
    double *d_out_params_tangents,
    cudaStream_t stream
) {

    // if(N != N_) {
    //     throw std::runtime_error("N != N_");
    // }

    // int tpb = 32;
    // int B = (N_+tpb-1)/tpb;

    // nblist_.compute_block_bounds(N_limit_, D, d_coords, stream);

    // gpuErrchk(cudaPeekAtLastError());

    // dim3 dimGrid(B, B, 1); // x, y, z dims
    // dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    // auto start = std::chrono::high_resolution_clock::now();
    // if(d_coords_tangents == nullptr) {

    //     // these can be ran in two streams later on
    //     k_nonbonded_inference<RealType, D><<<dimGrid, tpb, 0, stream>>>(
    //         N_limit_,
    //         d_coords,
    //         d_params,
    //         d_charge_param_idxs_,
    //         d_lj_param_idxs_,
    //         cutoff_,
    //         nblist_.get_block_bounds_ctr(),
    //         nblist_.get_block_bounds_ext(),
    //         d_out_coords
    //     );

    //     // cudaDeviceSynchronize();
    //     gpuErrchk(cudaPeekAtLastError());

    //     if(E_ > 0) {
    //         k_nonbonded_exclusion_inference<RealType, D><<<dimGridExclusions, tpb, 0, stream>>>(
    //             E_,
    //             d_coords,
    //             d_params,
    //             d_exclusion_idxs_,
    //             d_charge_scale_idxs_,
    //             d_lj_scale_idxs_,
    //             d_charge_param_idxs_,
    //             d_lj_param_idxs_,
    //             cutoff_,
    //             d_out_coords
    //         );
    //         // cudaDeviceSynchronize();
    //         gpuErrchk(cudaPeekAtLastError());
    //     }


    //     // cudaDeviceSynchronize();
    //     gpuErrchk(cudaPeekAtLastError());

    //     // auto finish = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double> elapsed = finish - start;
    //     // std::cout << "Nonbonded Elapsed time: " << elapsed.count() << " s\n";

    // } else {

    //     // do *not* accumulate tangents here
    //     k_nonbonded_jvp<RealType, D><<<dimGrid, tpb, 0, stream>>>(
    //         N_limit_,
    //         d_coords,
    //         d_coords_tangents,
    //         d_params,
    //         d_charge_param_idxs_,
    //         d_lj_param_idxs_,
    //         cutoff_,
    //         nblist_.get_block_bounds_ctr(),
    //         nblist_.get_block_bounds_ext(),
    //         d_out_coords_tangents,
    //         d_out_params_tangents
    //     );

    //     // cudaDeviceSynchronize();
    //     gpuErrchk(cudaPeekAtLastError());

    //     if(E_ > 0) {
    //         k_nonbonded_exclusion_jvp<RealType, D><<<dimGridExclusions, tpb, 0, stream>>>(
    //             E_,
    //             d_coords,
    //             d_coords_tangents,
    //             d_params,
    //             d_exclusion_idxs_,
    //             d_charge_scale_idxs_,
    //             d_lj_scale_idxs_,
    //             d_charge_param_idxs_,
    //             d_lj_param_idxs_,
    //             cutoff_,
    //             d_out_coords_tangents,
    //             d_out_params_tangents
    //         );            

    //         // cudaDeviceSynchronize();
    //         gpuErrchk(cudaPeekAtLastError());
    //     }


    //     // auto finish = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double> elapsed = finish - start;
    //     // std::cout << "Nonbonded JVP Elapsed time: " << elapsed.count() << " s\n";

    // }


};

template class Nonbonded<double, 4>;
template class Nonbonded<double, 3>;

template class Nonbonded<float, 4>;
template class Nonbonded<float, 3>;

} // namespace timemachine