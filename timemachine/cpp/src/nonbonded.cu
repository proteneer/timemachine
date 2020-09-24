#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include "nonbonded.hpp"
#include "hilbert.h"
#include "gpu_utils.cuh"

#include "k_nonbonded.cuh"

namespace timemachine {

template <typename RealType>
Nonbonded<RealType>::Nonbonded(
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &scales, // [E, 2]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double beta,
    double cutoff
) :  N_(lambda_offset_idxs.size()),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_offset_idxs.size()),
    beta_(beta),
    sort_counter_(0) {



    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(scales.size()/2 != E_) {
        throw std::runtime_error("bad scales size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, N_*sizeof(*d_du_dl_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, N_*sizeof(*d_u_buffer_)));
    gpuErrchk(cudaMalloc(&d_perm_, N_*sizeof(*d_perm_)));

    gpuErrchk(cudaMalloc(&d_sorted_lambda_offset_idxs_, N_*sizeof(*d_sorted_lambda_offset_idxs_)));
    gpuErrchk(cudaMalloc(&d_sorted_x_, N_*3*sizeof(*d_sorted_x_)));
    gpuErrchk(cudaMalloc(&d_sorted_p_, N_*3*sizeof(*d_sorted_p_)));
    gpuErrchk(cudaMalloc(&d_sorted_du_dx_, N_*3*sizeof(*d_sorted_du_dx_)));
    gpuErrchk(cudaMalloc(&d_sorted_du_dp_, N_*3*sizeof(*d_sorted_du_dp_)));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_scales_, E_*2*sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], E_*2*sizeof(*d_scales_), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMallocHost(&p_ixn_count_, 1*sizeof(*p_ixn_count_)));
    gpuErrchk(cudaMallocHost(&p_coords_, N_*3*sizeof(*p_coords_)));
    gpuErrchk(cudaMallocHost(&p_perm_, N_*sizeof(*p_perm_)));
    gpuErrchk(cudaMallocHost(&p_box_, 3*3*sizeof(*p_box_)));

};

template <typename RealType>
void Nonbonded<RealType>::hilbert_sort(cudaStream_t stream) {

    double bx = p_box_[0*3+0];
    double by = p_box_[1*3+1];
    double bz = p_box_[2*3+2];

    // 2. apply periodic centering
    for(int i=0; i < N_; i++) {

        double x = p_coords_[i*3+0];
        double y = p_coords_[i*3+1];
        double z = p_coords_[i*3+2];

        x -= bx*floor(x/bx);
        y -= by*floor(y/by);
        z -= bz*floor(z/bz);

        p_coords_[i*3+0] = x;
        p_coords_[i*3+1] = y;
        p_coords_[i*3+2] = z;
    }
    
    // 3. build the hilbert curve
    // if periodic
    double minx = 0.0;
    double miny = 0.0;
    double minz = 0.0;
    double maxx = bx;
    double maxy = by;
    double maxz = bz;

    double binWidth = max(max(maxx-minx, maxy-miny), maxz-minz)/255.0;
    double invBinWidth = 1.0/binWidth;
    std::vector<std::pair<int, int> > molBins(N_);

    for(int i = 0; i < N_; i++) {
        int x = (p_coords_[i*3+0]-minx)*invBinWidth;
        int y = (p_coords_[i*3+1]-miny)*invBinWidth;
        int z = (p_coords_[i*3+2]-minz)*invBinWidth;

        bitmask_t hilbert_coords[3];
        hilbert_coords[0] = x;
        hilbert_coords[1] = y;
        hilbert_coords[2] = z;
        int bin = (int) hilbert_c2i(3, 8, hilbert_coords);

        molBins[i] = std::pair<int, int>(bin, i);
    }

    std::sort(molBins.begin(), molBins.end());

    // 4. generate a new ordering
    for(int i=0; i < N_; i++) {
	// std::cout << "SORT: " << i << " " <<  molBins[i].second << std::endl;
        p_perm_[i] = molBins[i].second;
        // p_perm_[i] = i;
    }

    gpuErrchk(cudaMemcpyAsync(
        d_perm_,
        p_perm_,
        N_*sizeof(*d_perm_),
        cudaMemcpyHostToDevice,
        stream));

}

template <typename RealType>
Nonbonded<RealType>::~Nonbonded() {

    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_perm_)); // nullptr if we never built nblist
    
    gpuErrchk(cudaFree(d_sorted_x_));
    gpuErrchk(cudaFree(d_sorted_p_));
    gpuErrchk(cudaFree(d_sorted_du_dx_));
    gpuErrchk(cudaFree(d_sorted_du_dp_));
    gpuErrchk(cudaFree(d_sorted_lambda_offset_idxs_));

    gpuErrchk(cudaFreeHost(p_ixn_count_));
    gpuErrchk(cudaFreeHost(p_coords_));
    gpuErrchk(cudaFreeHost(p_box_));
    gpuErrchk(cudaFreeHost(p_perm_));

};

template <typename RealType>
void Nonbonded<RealType>::execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p, // N*3
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream) {

    assert(N == N_);
    assert(P == N_*3);

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    int sort_freq = 1; 

    const int B = (N+32-1)/32;
    const int tpb = 32;
    // sort atoms based on hilbert curve
    if(sort_counter_ % sort_freq == 0) {
        std::cout << "sorting" << std::endl;
        // copy data into pinnned buffers
        gpuErrchk(cudaMemcpyAsync(p_coords_, d_x, N*3*sizeof(*d_x), cudaMemcpyDeviceToHost))
        gpuErrchk(cudaMemcpyAsync(p_box_, d_box, 3*3*sizeof(*d_box), cudaMemcpyDeviceToHost))
        gpuErrchk(cudaStreamSynchronize(stream));
        this->hilbert_sort(stream);

	dim3 dimGrid(B, 3, 1);
        // params are N,3
        k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_p, d_sorted_p_);
        gpuErrchk(cudaPeekAtLastError());
    
        // lambda_idxs are N,1
        k_permute<<<B, tpb, 0, stream>>>(N, d_perm_, d_lambda_offset_idxs_, d_sorted_lambda_offset_idxs_);
        gpuErrchk(cudaPeekAtLastError());
    }

    sort_counter_ += 1;

    // sort coords, parameters
    dim3 dimGrid(B, 3, 1);

    // coords are N,3
    k_permute<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_x, d_sorted_x_);
    gpuErrchk(cudaPeekAtLastError());


    nblist_.build_nblist_device(
        N,
        d_sorted_x_,
        d_box,
        cutoff_,
        stream
    );

    gpuErrchk(cudaMemcpyAsync(p_ixn_count_, nblist_.get_ixn_count(), 1*sizeof(*p_ixn_count_), cudaMemcpyDeviceToHost, stream));
    // this stream needs to be synchronized so we can be sure that p_ixn_count_ is properly set.
    gpuErrchk(cudaStreamSynchronize(stream));

    // reset buffers and sorted accumulators
    if(d_du_dx) {
	gpuErrchk(cudaMemsetAsync(d_sorted_du_dx_, 0, N*3*sizeof(*d_sorted_du_dx_)))
    }
    if(d_du_dp) {
	gpuErrchk(cudaMemsetAsync(d_sorted_du_dp_, 0, N*3*sizeof(*d_sorted_du_dp_)))
    }

    if(d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, N*sizeof(*d_du_dl_buffer_), stream));        
    }
    if(d_u) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N*sizeof(*d_du_dl_buffer_), stream));        
    }

    k_nonbonded<RealType><<<p_ixn_count_[0], 32, 0, stream>>>(
        N,
        d_sorted_x_,
        d_sorted_p_,
        d_box,
        lambda,
        d_sorted_lambda_offset_idxs_,
        beta_,
        cutoff_,
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_du_dx ? d_sorted_du_dx_ : nullptr,
        d_du_dp ? d_sorted_du_dp_ : nullptr,
        d_du_dl ? d_du_dl_buffer_ : nullptr, // switch to nullptr if we don't request du_dl
        d_u ? d_u_buffer_ : nullptr // switch to nullptr if we don't request energies
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // copy over tiled ixns
    // coords are N,3
    if(d_du_dx) {
        k_inv_permute_accum<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_sorted_du_dx_, d_du_dx);
        gpuErrchk(cudaPeekAtLastError());
    }

    // params are N,3
    // this needs to be an accumlated permute
    if(d_du_dp) {
        k_inv_permute_accum<<<dimGrid, tpb, 0, stream>>>(N, d_perm_, d_sorted_du_dp_, d_du_dp);
        gpuErrchk(cudaPeekAtLastError());
    }

    // exclusions use the non-sorted version
    if(E_ > 0) {

        const int tpb = 32;
        dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

        k_nonbonded_exclusions<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_x,
            d_p,
            d_box,
            lambda,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_, // FIX EXCLUSIONS
            d_scales_, // FIX SCALES
            beta_,
            cutoff_,
            d_du_dx,
            d_du_dp,
            d_du_dl ? d_du_dl_buffer_ : nullptr, // switch to nullptr if we don't request du_dl
            d_u ? d_u_buffer_ : nullptr // switch to nullptr if we don't request energies
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_du_dl) {
        k_reduce_buffer<<<B, 32, 0, stream>>>(N, d_du_dl_buffer_, d_du_dl);
        gpuErrchk(cudaPeekAtLastError());
    }

    if(d_u) {
        k_reduce_buffer<<<B, 32, 0, stream>>>(N, d_u_buffer_, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
    
}

template class Nonbonded<double>;
template class Nonbonded<float>;

} // namespace timemachine
