#include <vector>
#include <iostream>

#include "neighborlist.hpp"
#include "k_find_block_bounds.cuh"
#include "gpu_utils.cuh"

namespace timemachine {

Neighborlist::Neighborlist(
	int N,
	int D) {

    int tpb = 32;
    int B = (N+tpb-1)/tpb;

    gpuErrchk(cudaMalloc(&d_block_bounds_ctr_, B*D*sizeof(*d_block_bounds_ctr_)));
    gpuErrchk(cudaMalloc(&d_block_bounds_ext_, B*D*sizeof(*d_block_bounds_ext_)));

}

Neighborlist::~Neighborlist() {

    gpuErrchk(cudaFree(d_block_bounds_ctr_));
    gpuErrchk(cudaFree(d_block_bounds_ext_));

}


bool is_pow_2(int x) {
    return (x & (x - 1)) == 0;
}

int log2_int(int v) {
    int bits = 0;
    while (v >>= 1) ++bits;
    return bits;
}

int pow_int(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = pow_int(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}


void Neighborlist::compute_block_bounds_cpu(
    const int N,
    const int D,
    const int block_size,
    const double *h_coords,
    const double *h_box,
    double *bb_ctrs,
    double *bb_exts) {
   

    int num_blocks = (N + block_size - 1)/block_size;

    for(int block_idx = 0; block_idx < num_blocks; block_idx++) {
        for(int d=0; d < D; d++) {
            double width = h_box[d*3+d];
            double ci_min =  9999999;
            double ci_max = -9999999;
            for(int i=0; i < block_size; i++) {
                int tid = block_idx*block_size + i;
                if(tid < N) {
                    // int atom_idx = perm[tid];
                    int atom_idx = tid;
                    double ci = h_coords[atom_idx*D + d];
                    ci -= width*floor(ci/width); // move to home box
                    ci_min = ci < ci_min ? ci : ci_min;
                    ci_max = ci > ci_max ? ci : ci_max;
                }
            }
         
            // printf("dim %d block_idx %d ctr %f ext %f\n", d, block_idx, (ci_max + ci_min)/2.0, ci_max - ci_min);
            bb_ctrs[block_idx*D+d] = (ci_max + ci_min)/2.0;
            bb_exts[block_idx*D+d] = (ci_max - ci_min)/2.0;
        }
    }

}

std::vector<std::vector<int> >  Neighborlist::build_nblist_cpu(
    int N,
    int D,
    const double *h_coords,
    const double *h_box,
    const double cutoff) {
    /*
    Reference neighborlist algorithm:
    Given N atoms, we subdivide into blocks of size 2^B log_2(N) times.


    Suppose we have 256 atoms. We compute block bounds in groups of
    128, 64, 32, 16, 8, 4, 2, 1. In the limit as the group size == 1 we have the
    coordinates themselves.

    We proceed by

    128x128
    64x64
    32x32
    32x16
    32x8
    32x4
    32x2
    32x1

    When we're finally down to 32x1 we do a full check
   
    */

    // std::cout << log2_int(31) << std::endl;
    // std::cout << log2_int(32) << std::endl;
    // std::cout << log2_int(33) << std::endl;

    // throw std::runtime_error("N is not a power of 2.");

    // if(!is_pow_2(N)) {
    //     throw std::runtime_error("N is not a power of 2.");
    // }

    int row_block_size = N;
    int col_block_size = N;

    // int debug = 0;

    // std::cout << "debug " << debug++ << std::endl;

    //
    // 2^0 = 1
    // 2^1 = 2
    // 2^2 = 4
    // 2^3 = 8
    // 2^4 = 16     
    // 2^5 = 32    num_rounds == 6
    int num_rounds = log2_int(N) + 1;
    num_rounds = max(num_rounds, 6); // need at least 6 rounds

    std::vector<std::vector<double> > all_block_ctrs;
    std::vector<std::vector<double> > all_block_exts;

    // compute block bounds t
    for(int round=0; round < num_rounds; round++)  {
        int block_size = pow_int(2, round);
        int num_blocks = (N + block_size - 1)/block_size;

        std::vector<double> bb_ctrs(num_blocks*3);
        std::vector<double> bb_exts(num_blocks*3);
        this->compute_block_bounds_cpu(
            N,
            D,
            block_size,
            h_coords,
            h_box,
            &bb_ctrs[0],
            &bb_exts[0]
        );

        all_block_ctrs.push_back(bb_ctrs);
        all_block_exts.push_back(bb_exts);

    }

    const int num_blocks_of_32 = (N + 32 - 1) / 32;

    const int bound_idx_32 = log2_int(32);
    // implement OpenMM's bbox approach

    std::vector<std::vector<int> > ixn_list;


    double bx = h_box[0*3+0];
    double by = h_box[1*3+1];
    double bz = h_box[2*3+2];

    // std::cout << bound_idx_32 << std::endl;
    // std::cout << num_blocks_of_32 << std::endl;
    // std::cout << all_block_ctrs.size() << std::endl;
    // std::cout << all_block_exts.size() << std::endl;

    // std::cout << "debug " << debug++ << std::endl;

    for(int rbidx=0; rbidx < num_blocks_of_32; rbidx++) {

        // std::cout << "debug " << debug++ << std::endl;

        double box_ctr_x = all_block_ctrs[bound_idx_32][rbidx*3+0];
        double box_ctr_y = all_block_ctrs[bound_idx_32][rbidx*3+1];
        double box_ctr_z = all_block_ctrs[bound_idx_32][rbidx*3+2];

        // std::cout << "debug " << debug++ << std::endl;

        double box_ext_x = all_block_exts[bound_idx_32][rbidx*3+0];
        double box_ext_y = all_block_exts[bound_idx_32][rbidx*3+1];
        double box_ext_z = all_block_exts[bound_idx_32][rbidx*3+2];


        // std::cout << "debug " << debug++ << std::endl;
        std::vector<int> interacting_idxs;

        for(int j=0; j < N; j++) {

            double jx = h_coords[j*3+0];
            double jy = h_coords[j*3+1];
            double jz = h_coords[j*3+2];

            double dx = box_ctr_x - jx;
            double dy = box_ctr_y - jy;
            double dz = box_ctr_z - jz;

            dx -= bx*floor(dx/bx+0.5);
            dy -= by*floor(dy/by+0.5);
            dz -= bz*floor(dz/bz+0.5);

            dx = max(0.0, fabs(dx) - box_ext_x);
            dy = max(0.0, fabs(dy) - box_ext_y);
            dz = max(0.0, fabs(dz) - box_ext_z);

            double box_dist = sqrt(dx*dx + dy*dy + dz*dz);


            int row_start = rbidx*32;
            int row_end = min((rbidx+1)*32, N);

            if(box_dist > cutoff) {
                continue;
            }

            bool keep = false;
            
            for(int i=row_start; i < row_end; i++) {

                double ix = h_coords[i*3+0];
                double iy = h_coords[i*3+1];
                double iz = h_coords[i*3+2];

                double dx = ix - jx;
                double dy = iy - jy;
                double dz = iz - jz;

                dx -= bx*floor(dx/bx+0.5);
                dy -= by*floor(dy/by+0.5);
                dz -= bz*floor(dz/bz+0.5);

                double atom_dist = sqrt(dx*dx + dy*dy + dz*dz);

                if(atom_dist < cutoff) {
                    keep = true;
                    break;
                }
            }

            if(keep) {
                interacting_idxs.push_back(j);
            }

        }

        ixn_list.push_back(interacting_idxs);

    }

    return ixn_list;

    // for(int row=0; row < num_rows; row++) {
    //     for(int col=0; col < num_cols; col++) {

    //     }
    // }
    // 
}

void Neighborlist::compute_block_bounds(
	int N,
	int D,
	const double *coords,
    const double *box,
    const int *perm,
	cudaStream_t stream) {

    int tpb = 32;
    int B = (N+tpb-1)/tpb; // total number of blocks we need to process

    gpuErrchk(cudaMemsetAsync(d_block_bounds_ctr_, 0, B*D*sizeof(*d_block_bounds_ctr_), stream));
    gpuErrchk(cudaMemsetAsync(d_block_bounds_ext_, 0, B*D*sizeof(*d_block_bounds_ext_), stream));

    k_find_block_bounds<<<B, tpb, 0, stream>>>(
        N,
        D,
        B,
        coords,
        box,
        perm,
        d_block_bounds_ctr_,
        d_block_bounds_ext_
    );	

    gpuErrchk(cudaPeekAtLastError());

};


}