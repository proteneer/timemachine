#include <chrono>
#include <iostream>

#include "fast_nonbonded.hpp"
#include "k_nonbonded_deterministic.cuh"

#include <vector>
#include "gpu_utils.cuh"
#include "surreal.cuh"

#include "assert.h"

template<typename RealType, int D>
void fast_nonbonded_normal(
    const RealType *coords,
    const RealType *coords_tangents,
    const RealType *params,
    const int *param_idxs,
    const double cutoff,
    int N,
    int P,
    RealType *out_coords,
    RealType *out_coords_tangents,
    RealType *out_params_tangents) {

    RealType *d_coords;
    RealType *d_params;

    int *d_param_idxs;

    int ND = N*D;

    gpuErrchk(cudaMalloc(&d_coords, sizeof(RealType)*ND));
    gpuErrchk(cudaMalloc(&d_params, sizeof(RealType)*P));
    gpuErrchk(cudaMalloc(&d_param_idxs, sizeof(int)*N));

    gpuErrchk(cudaMemcpy(d_coords, coords, sizeof(RealType)*ND, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, params, sizeof(RealType)*P, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs, param_idxs, sizeof(int)*N, cudaMemcpyHostToDevice));


    int tpb = 32;
    int n_blocks = (N+tpb-1)/tpb;

    RealType *block_bounds_ctr;
    RealType *block_bounds_ext;

    gpuErrchk(cudaMalloc(&block_bounds_ctr, sizeof(*block_bounds_ctr)*n_blocks*D));
    gpuErrchk(cudaMalloc(&block_bounds_ext, sizeof(*block_bounds_ctr)*n_blocks*D));
    gpuErrchk(cudaMemset(block_bounds_ctr, 0, sizeof(*block_bounds_ctr)*n_blocks*D));
    gpuErrchk(cudaMemset(block_bounds_ext, 0, sizeof(*block_bounds_ctr)*n_blocks*D));

    k_find_block_bounds<RealType><<<1, n_blocks>>>(
        N,
        D,
        n_blocks,
        d_coords,
        block_bounds_ctr,
        block_bounds_ext
    );

    cudaDeviceSynchronize();

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(n_blocks, n_blocks, 1); // x, y, z dims

    if(coords_tangents == nullptr) {

        assert(out_coords != nullptr);
        assert(out_coords_tangents == nullptr);
        assert(out_params_tangents == nullptr);

        unsigned long long *d_out_coords;
        gpuErrchk(cudaMalloc(&d_out_coords, sizeof(unsigned long long)*ND));
        gpuErrchk(cudaMemset(d_out_coords, 0, sizeof(unsigned long long)*ND));

        auto start = std::chrono::high_resolution_clock::now();
        k_nonbonded_inference<RealType, D><<<dimGrid, tpb>>>(
            N,
            d_coords,
            d_params,
            d_param_idxs,
            cutoff,
            block_bounds_ctr,
            block_bounds_ext,
            d_out_coords
        );

        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Nonbonded Elapsed time: " << elapsed.count() << " s\n";

        std::vector<unsigned long long> h_host_coords(N*D);
        gpuErrchk(cudaMemcpy(&h_host_coords[0], d_out_coords, sizeof(unsigned long long)*ND, cudaMemcpyDeviceToHost));
        for(int i=0; i<h_host_coords.size(); i++) {
            out_coords[i] = static_cast<RealType>(static_cast<signed long long>(h_host_coords[i]))/0x100000000;;
        }

    } else {

        assert(out_coords == nullptr);
        assert(out_coords_tangents != nullptr);
        assert(out_params_tangents != nullptr);

        RealType *d_in_coords_tangents;
        RealType *d_out_coords_tangents;
        RealType *d_out_params_tangents;

        gpuErrchk(cudaMalloc(&d_in_coords_tangents, sizeof(RealType)*ND));
        gpuErrchk(cudaMalloc(&d_out_coords_tangents, sizeof(RealType)*ND));
        gpuErrchk(cudaMalloc(&d_out_params_tangents, sizeof(RealType)*ND));

        gpuErrchk(cudaMemcpy(d_in_coords_tangents, coords_tangents, sizeof(RealType)*ND, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(d_out_coords_tangents, 0, sizeof(RealType)*ND));
        gpuErrchk(cudaMemset(d_out_params_tangents, 0, sizeof(RealType)*P));


        auto start = std::chrono::high_resolution_clock::now();
        k_nonbonded_jvp<RealType, D><<<dimGrid, tpb>>>(
            N,
            d_coords,
            d_in_coords_tangents,
            d_params,
            d_param_idxs,
            cutoff,
            block_bounds_ctr,
            block_bounds_ext,
            d_out_coords_tangents,
            d_out_params_tangents
        );

        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Nonbonded JVP Elapsed time: " << elapsed.count() << " s\n";

        gpuErrchk(cudaMemcpy(out_coords_tangents, d_out_coords_tangents, sizeof(RealType)*ND, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(out_params_tangents, d_out_params_tangents, sizeof(RealType)*P, cudaMemcpyDeviceToHost));

    }

}

template void fast_nonbonded_normal<double, 3>(
    const double *coords,
    const double *coords_tangents,
    const double *params,
    const int *param_idxs,
    const double cutoff,
    int N,
    int P,
    double *out_coords,
    double *out_coords_tangents,
    double *out_params_tangents);


template void fast_nonbonded_normal<double, 4>(
    const double *coords,
    const double *coords_tangents,
    const double *params,
    const int *param_idxs,
    const double cutoff,
    int N,
    int P,
    double *out_coords,
    double *out_coords_tangents,
    double *out_params_tangents);
