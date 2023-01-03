// Kernels specific to Local MD implementation.

void __global__ k_construct_bonded_params(
    const int num_idxs,               // Number of idxs
    const int num_atoms,              // Max value any idx can be
    const unsigned int reference_idx, // Atom index to create bonds to
    const double k,
    const double r_min,
    const double r_max,
    const unsigned int *__restrict__ idxs, // [num_idxs]
    int *__restrict__ bonds,               // [num_idxs * 2]
    double *__restrict__ params            // [num_idxs * 3]
) {
    const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_idxs) {
        return;
    }
    const unsigned int atom_idx = idxs[idx];
    if (atom_idx >= num_atoms) {
        return;
    }
    bonds[idx * 2 + 0] = reference_idx;
    bonds[idx * 2 + 1] = atom_idx;

    params[idx * 3 + 0] = k;
    params[idx * 3 + 1] = r_min;
    params[idx * 3 + 2] = r_max;
}
