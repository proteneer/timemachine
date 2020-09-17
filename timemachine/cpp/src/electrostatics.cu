#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include <sstream>
#include <numeric>
#include <algorithm>

#include "hilbert.h"
#include "electrostatics.hpp"
#include "gpu_utils.cuh"

#include "k_electrostatics.cuh"
#include "k_electrostatics_jvp.cuh"

namespace timemachine {

template <typename RealType>
Electrostatics<RealType>::Electrostatics(
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &charge_scales, // [E]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double beta,
    double cutoff
) :  N_(lambda_offset_idxs.size()),
    beta_(beta),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_offset_idxs.size(), 3),
    d_perm_(nullptr) {

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(charge_scales.size()*2 != exclusion_idxs.size()) {
        throw std::runtime_error("charge scale idxs size not half of exclusion size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_scales_, E_*sizeof(*d_charge_scales_)));
    gpuErrchk(cudaMemcpy(d_charge_scales_, &charge_scales[0], E_*sizeof(*d_charge_scales_), cudaMemcpyHostToDevice));


};

template <typename RealType>
Electrostatics<RealType>::~Electrostatics() {
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_charge_scales_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
    gpuErrchk(cudaFree(d_perm_));
};

struct Vec3 {

    double x, y, z;

    Vec3(double x, double y, double z) : x(x), y(y), z(z) {};

    Vec3 operator*(double a) {
        return Vec3(x*a, y*a, z*a);
    }


};

template <typename RealType>
void Electrostatics<RealType>::execute_device(
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

    int WSIZE = 32;
    const int B = (N_+WSIZE-1)/WSIZE;
    const int D = 3;



    // add conditional for flipping a random number.
    bool sort = 0;
    if(d_perm_ == nullptr) {
        gpuErrchk(cudaMalloc(&d_perm_, N_*sizeof(*d_perm_)));
        sort = true;
    }

    // we will add a predicate that re-shuffles once every 30 steps or so on average

    auto start_sort = std::chrono::high_resolution_clock::now();

    // can probably be turned into a float for speed, since the nblist is approximate anyways


    std::vector<double> centered_coords(N_*3);
    std::vector<int> perm(N_);


    // tbd switch with asyncversions
    std::vector<double> box(9);
    gpuErrchk(cudaMemcpy(&box[0], d_box, 9*sizeof(double), cudaMemcpyDeviceToHost));

    // 1. copy over coordinates
    std::vector<double> coords(N_*3);
    gpuErrchk(cudaMemcpy(&coords[0], d_x, N_*3*sizeof(double), cudaMemcpyDeviceToHost));
 
    // nblist_.build_nblist_cpu(
    //     N,
    //     D,
    //     &coords[0]
    // );

    throw std::runtime_error("break");


    if(sort) {
        double bx = box[0*3+0];
        double by = box[1*3+1];
        double bz = box[2*3+2];

        // 2. apply periodic centering
        for(int i=0; i < N_; i++) {

            double x = coords[i*3+0];
            double y = coords[i*3+1];
            double z = coords[i*3+2];

            // only if periodic
            x -= bx*floor(x/bx);
            y -= by*floor(y/by);
            z -= bz*floor(z/bz);

            centered_coords[i*3+0] = x;
            centered_coords[i*3+1] = y;
            centered_coords[i*3+2] = z;
        }
        
        // 3. build the hilbert curve
        // if periodic
        // double minx = 0.0;
        // double miny = 0.0;
        // double minz = 0.0;
        // double maxx = bx;
        // double maxy = by;
        // double maxz = bz;

        // always use this to generate the bounding box
        double minx = centered_coords[0*3+0], maxx = centered_coords[0*3+0];
        double miny = centered_coords[0*3+1], maxy = centered_coords[0*3+1];
        double minz = centered_coords[0*3+2], maxz = centered_coords[0*3+2];
        for (int i = 1; i < N_; i++) {
            // const Real4& pos = oldPosq[i];
            minx = min(minx, centered_coords[i*3+0]);
            maxx = max(maxx, centered_coords[i*3+0]);
            miny = min(miny, centered_coords[i*3+1]);
            maxy = max(maxy, centered_coords[i*3+1]);
            minz = min(minz, centered_coords[i*3+2]);
            maxz = max(maxz, centered_coords[i*3+2]);
        }

        double binWidth = max(max(maxx-minx, maxy-miny), maxz-minz)/255.0;
        double invBinWidth = 1.0/binWidth;
        std::vector<std::pair<int, int> > molBins(N_);

        for(int i = 0; i < N_; i++) {
            int x = (centered_coords[i*3+0]-minx)*invBinWidth;
            int y = (centered_coords[i*3+1]-miny)*invBinWidth;
            int z = (centered_coords[i*3+2]-minz)*invBinWidth;

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
            // perm[i] = i;
            perm[i] = molBins[i].second;
        }
        gpuErrchk(cudaMemcpy(d_perm_, &perm[0], N*sizeof(*d_perm_), cudaMemcpyHostToDevice));
    }

    auto end_sort = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_sort - start_sort).count();
    std::cout << duration << "us to re-sort" << std::endl;;

    // its safe for us to build a neighborlist in a lower dimension.
    // nblist_.compute_block_bounds(
    //     N_,
    //     D,
    //     d_x,
    //     d_box,
    //     // d_perm_,
    //     stream
    // );

    std::vector<double> bb_ctr(B*3);
    std::vector<double> bb_ext(B*3);

    // gpuErrchk(cudaMemcpy(&bb_ctr[0], nblist_.get_block_bounds_ctr(), B*3*sizeof(double), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(&bb_ext[0], nblist_.get_block_bounds_ext(), B*3*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<int> tiles_x;
    std::vector<int> tiles_y;

    // std::vector<double> box(9);
    // gpuErrchk(cudaMemcpy(&box[0], d_box, 9*sizeof(double), cudaMemcpyDeviceToHost));

    double bx[3] = {box[0*3+0], box[1*3+1], box[2*3+2]};

    std::vector<double> bv;


    for(int x=0; x < B; x++) {

        double vol = 1;

        for(int d=0; d < 3; d++) {   
            double block_row_ext = bb_ext[x*3+d];
            vol *= 2*block_row_ext*2*block_row_ext;
        }
        bv.push_back(vol);

    }

    double box_sum = std::accumulate(bv.begin(), bv.end(), 0.0);
    double box_mean = box_sum / bv.size();

    double box_sq_sum = std::inner_product(bv.begin(), bv.end(), bv.begin(), 0.0);
    double box_stdev = std::sqrt(box_sq_sum / bv.size() - box_mean * box_mean);

    std::cout << "box mean " << box_mean << " std " << box_stdev << std::endl;

    // throw std::runtime_error("debug");


    // check bounding box deltas
    for(int x=0; x < B; x++) {

        for(int y=0; y < B; y++) {

            if(y > x) {
                continue;
            }

            double block_d2ij = 0;
 
            for(int d=0; d < 3; d++) {
                double block_row_ctr = bb_ctr[x*3+d];
                double block_row_ext = bb_ext[x*3+d];
                double block_col_ctr = bb_ctr[y*3+d];
                double block_col_ext = bb_ext[y*3+d];

                double dx = block_row_ctr - block_col_ctr;
                dx -= bx[d]*floor(dx/bx[d]+static_cast<double>(0.5));
                dx = max(static_cast<double>(0.0), fabs(dx) - (block_row_ext + block_col_ext));
                block_d2ij += dx*dx;                
            }

            if(block_d2ij < cutoff_*cutoff_) {
                tiles_x.push_back(x);
                tiles_y.push_back(y);
            } else{
                // std::cout << "skipping: " << x << " " << y << std::endl;
            }

        }
    }

    int *d_tiles_x_ = gpuErrchkCudaMallocAndCopy(&tiles_x[0], tiles_x.size());
    int *d_tiles_y_ = gpuErrchkCudaMallocAndCopy(&tiles_y[0], tiles_y.size());

    int ref_total_ixns = 0;

    // test fully compact tiles


    std::vector<int> interacting_y_offsets;
    std::vector<int> all_interacting_ys;


    int compact_tile_size = 0;
    for(int bi=0; bi < B; bi++) {

        std::vector<int> interacting_ys;
        std::vector<int> interacting_y_counts;

        for(int j=0; j < N_; j++) {

            // skip upper right triangular parts
            if(j > bi*32) {
                continue;
            }

            double cj[3] = {
                coords[perm[j]*3+0],
                coords[perm[j]*3+1],
                coords[perm[j]*3+2]
            };

            int j_ixns = 0;

            for(int i = bi*32; i < min((bi+1)*32, N_); i++) {

                double ci[3] = {
                    coords[perm[i]*3+0],
                    coords[perm[i]*3+1],
                    coords[perm[i]*3+2]
                };

                double dij = 0;

                for(int d=0; d < 3; d++) {
                    double delta = ci[d] - cj[d];
                    delta -= box[d*3+d]*floor(delta/box[d*3+d]+0.5);
                    dij += delta*delta;                
                }

                dij = sqrt(dij);

                if(dij < cutoff_) {
                    j_ixns += 1;
                }

            }

            if(j_ixns > 0) {
                interacting_ys.push_back(j);
                interacting_y_counts.push_back(j_ixns);
            }

            ref_total_ixns += j_ixns;

        }

        interacting_y_offsets.push_back(interacting_ys.size()); // needs to get prefix summ'd

        for(int j=0; j < interacting_ys.size(); j++) {
            all_interacting_ys.push_back(interacting_ys[j]);
        }

        interacting_y_offsets.push_back(interacting_ys.size());

        compact_tile_size += interacting_ys.size()/32;

        // std::cout << "block " << bi << " interacts with " << interacting_ys.size() << " neighbors " << std::endl;

        // sort by y ixn counts
        // if(bi == 500) {
        //     std::sort(interacting_y_counts.begin(), interacting_y_counts.end());

        //     for(int q=0; q < interacting_y_counts.size(); q++) {
        //         std::cout << "q: " << interacting_y_counts[q] << std::endl; 
        //     }

        //     throw std::runtime_error("debug");
        // }


    }

    std::cout << "compact tile size: " << compact_tile_size << std::endl; 
    // throw std::runtime_error("debug");


    std::cout << "num_tiles: " << tiles_x.size() << " out of " << (N_/32)*(N_/32) << std::endl;

    gpuErrchk(cudaPeekAtLastError());

    // remove me later
    cudaDeviceSynchronize();



    auto start = std::chrono::high_resolution_clock::now();


    // these can be ran in two streams later on
    int *total_ixns;
    gpuErrchk(cudaMallocManaged(&total_ixns, 1*sizeof(int)));

    int *total_empty_tiles;
    gpuErrchk(cudaMallocManaged(&total_empty_tiles, 1*sizeof(int)));

    // const int TILES = (tiles_x.size()+tpb-1)/tpb;
    const int TILES = tiles_x.size();

    std::cout << "NUM TILES: " << TILES << std::endl;

    const int tpb = 32;
    const int div = tpb/32;
    const int NBLOCKS = (TILES + div - 1)/div;

    // dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGrid(NBLOCKS, 1, 1); // x, y, z dims

    k_electrostatics<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        TILES,
        d_x,
        d_p,
        d_box,
        lambda,
        d_lambda_offset_idxs_,
        beta_,
        cutoff_,
        // nblist_.get_block_bounds_ctr(),
        // nblist_.get_block_bounds_ext(),
        d_tiles_x_,
        d_tiles_y_,
        d_perm_,
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u,
        total_ixns,
        total_empty_tiles
    );

    cudaDeviceSynchronize();

    std::cout << "total ixns: " << *total_ixns << "/" << TILES*(32*32) << std::endl;

    std::cout << "compact ixns: " << *total_ixns << "/" << compact_tile_size*32*32 << std::endl; 

    std::cout << "total empty tiles: " << *total_empty_tiles << "/" << tiles_x.size() << std::endl;

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if(E_ > 0) {


        dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

        // k_electrostatics_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
        //     E_,
        //     d_x,
        //     d_p,
        //     d_box,
        //     lambda,
        //     d_lambda_offset_idxs_,
        //     d_exclusion_idxs_,
        //     d_charge_scales_,
        //     beta_,
        //     cutoff_,
        //     d_du_dx,
        //     d_du_dp,
        //     d_du_dl,
        //     d_u
        // );
        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
}

template class Electrostatics<double>;
template class Electrostatics<float>;

} // namespace timemachine