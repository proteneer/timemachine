#include "k_nonbonded.cuh"

void __global__ k_coords_to_kv(
    const int N,
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals) {

    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom_idx >= N) {
        return;
    }

    // these coords have to be centered
    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    double binWidth = max(max(bx, by), bz) / 255.0;

    double x = coords[atom_idx * 3 + 0];
    double y = coords[atom_idx * 3 + 1];
    double z = coords[atom_idx * 3 + 2];

    x -= bx * floor(x / bx);
    y -= by * floor(y / by);
    z -= bz * floor(z / bz);

    unsigned int bin_x = x / binWidth;
    unsigned int bin_y = y / binWidth;
    unsigned int bin_z = z / binWidth;

    keys[atom_idx] = bin_to_idx[bin_x * 256 * 256 + bin_y * 256 + bin_z];
    // uncomment below if you want to preserve the atom ordering
    // keys[atom_idx] = atom_idx;
    vals[atom_idx] = atom_idx;
}

// TODO: DRY with k_coords_to_kv
void __global__ k_coords_to_kv_gather(
    const int N,
    const unsigned int *atom_idxs,
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    const int atom_idx = atom_idxs[idx];

    // these coords have to be centered
    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    double binWidth = max(max(bx, by), bz) / 255.0;

    double x = coords[atom_idx * 3 + 0];
    double y = coords[atom_idx * 3 + 1];
    double z = coords[atom_idx * 3 + 2];

    x -= bx * floor(x / bx);
    y -= by * floor(y / by);
    z -= bz * floor(z / bz);

    unsigned int bin_x = x / binWidth;
    unsigned int bin_y = y / binWidth;
    unsigned int bin_z = z / binWidth;

    keys[idx] = bin_to_idx[bin_x * 256 * 256 + bin_y * 256 + bin_z];
    // uncomment below if you want to preserve the atom ordering
    // keys[idx] = atom_idx;
    vals[idx] = atom_idx;
}

void __global__ k_arange(int N, unsigned int *arr) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= N) {
        return;
    }
    arr[atom_idx] = atom_idx;
}

// From https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expand_bits(unsigned int v) {
    // TODO: rewrite bit magic in more literate style
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// From https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1]^3.
template <typename RealType> unsigned int morton_encode(RealType x, RealType y, RealType z) {
    // TODO: rewrite without 3x repetition
    float x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    float y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    float z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}
