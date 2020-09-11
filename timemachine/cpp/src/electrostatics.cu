#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include <sstream>
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

    const int tpb = 32;
    const int B = (N_+tpb-1)/tpb;
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
    if(sort) {
        // tbd switch with asyncversions
        std::vector<double> box(9);
        gpuErrchk(cudaMemcpy(&box[0], d_box, 9*sizeof(double), cudaMemcpyDeviceToHost));

        // 1. copy over coordinates
        std::vector<double> coords(N_*3);
        gpuErrchk(cudaMemcpy(&coords[0], d_x, N_*3*sizeof(double), cudaMemcpyDeviceToHost));
     
        std::vector<double> centered_coords(N_*3);

        double bx = box[0*3+0];
        double by = box[1*3+1];
        double bz = box[2*3+2];

        // 2. apply periodic centering
        for(int i=0; i < N_; i++) {

            double x = coords[i*3+0];
            double y = coords[i*3+1];
            double z = coords[i*3+2];

            z -= bz*floor(z/bz);
            y -= by*floor(y/by);
            x -= bx*floor(x/bx);

            centered_coords[i*3+0] = x;
            centered_coords[i*3+1] = y;
            centered_coords[i*3+2] = z;
        }
        
        // 3. build the hilbert curve
        // if periodic
        double minx = 0.0;
        double miny = 0.0;
        double minz = 0.0;
        double maxx = bx;
        double maxy = by;
        double maxz = bz;

        // if not periodic
        // double minx = coords[0*3+0], maxx = coords[0*3+0];
        // double miny = coords[0*3+1], maxy = coords[0*3+1];
        // double minz = coords[0*3+2], maxz = coords[0*3+2];
        // for (int i = 1; i < N_; i++) {
        //     // const Real4& pos = oldPosq[i];
        //     minx = min(minx, coords[i*3+0]);
        //     maxx = max(maxx, coords[i*3+0]);
        //     miny = min(miny, coords[i*3+1]);
        //     maxy = max(maxy, coords[i*3+1]);
        //     minz = min(minz, coords[i*3+2]);
        //     maxz = max(maxz, coords[i*3+2]);
        // }
        
        double binWidth = max(max(maxx-minx, maxy-miny), maxz-minz)/255.0;
        double invBinWidth = 1.0/binWidth;
        std::vector<std::pair<int, int> > molBins(N_);
        bitmask_t hilbert_coords[3];
        for(int i = 0; i < N_; i++) {
            int x = (centered_coords[i*3+0]-minx)*invBinWidth;
            int y = (centered_coords[i*3+1]-miny)*invBinWidth;
            int z = (centered_coords[i*3+2]-minz)*invBinWidth;

            hilbert_coords[0] = x;
            hilbert_coords[1] = y;
            hilbert_coords[2] = z;
            int bin = (int) hilbert_c2i(3, 8, hilbert_coords);

            molBins[i] = std::pair<int, int>(bin, i);
        }
        std::sort(molBins.begin(), molBins.end());
        // 4. generate a new ordering
        std::vector<int> perm(N_);
        for(int i=0; i < N_; i++) {
            perm[i] = molBins[i].second;
        }
        gpuErrchk(cudaMemcpy(d_perm_, &perm[0], N*sizeof(*d_perm_), cudaMemcpyHostToDevice));
    }

    auto end_sort = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_sort - start_sort).count();
    std::cout << duration << "us to resort" << std::endl;;

    // its safe for us to build a neighborlist in a lower dimension.
    nblist_.compute_block_bounds(
        N_,
        D,
        d_x,
        d_box,
        d_perm_,
        stream
    );

    gpuErrchk(cudaPeekAtLastError());

    // remove me
    cudaDeviceSynchronize();

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();


    // these can be ran in two streams later on
    k_electrostatics<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_x,
        d_p,
        d_box,
        lambda,
        d_lambda_offset_idxs_,
        beta_,
        cutoff_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_perm_,
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if(E_ > 0) {
        k_electrostatics_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_x,
            d_p,
            d_box,
            lambda,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_charge_scales_,
            beta_,
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

template class Electrostatics<double>;
template class Electrostatics<float>;

} // namespace timemachine