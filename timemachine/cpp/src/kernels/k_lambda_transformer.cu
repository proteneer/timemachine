#include "k_lambda_transformer.cuh"

void __global__ k_compute_w_coords(
    const int N,
    const double lambda,
    const double cutoff,
    const int *__restrict__ lambda_plane_idxs, // 0 or 1, shift
    const int *__restrict__ lambda_offset_idxs,
    double *__restrict__ coords_w) {

    int atom_i_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom_i_idx >= N) {
        return;
    }

    int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;
    int lambda_plane_i = atom_i_idx < N ? lambda_plane_idxs[atom_i_idx] : 0;
    double f_lambda = transform_lambda_w(lambda);
    double coords_w_i = (lambda_plane_i + lambda_offset_i * f_lambda) * cutoff;
    coords_w[atom_i_idx] = coords_w_i;

} // 0 or 1, how much we offset from the plane by )
