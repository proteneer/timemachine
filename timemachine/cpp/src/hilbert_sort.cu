#include "gpu_utils.cuh"
#include "hilbert_sort.hpp"
#include "kernels/k_hilbert.cuh"
#include "vendored/hilbert.h"
#include <cub/cub.cuh>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace timemachine {

HilbertSort::HilbertSort(const int N)
    : N_(N), d_bin_to_idx_(HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM), d_sort_keys_in_(N),
      d_sort_keys_out_(N), d_sort_vals_in_(N), d_sort_storage_(0), d_sort_storage_bytes_(0) {
    // initialize hilbert curve which maps each of the HILBERT_GRID_DIM x HILBERT_GRID_DIM x HILBERT_GRID_DIM cells into an index.
    std::vector<unsigned int> bin_to_idx(HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM);
    for (int i = 0; i < HILBERT_GRID_DIM; i++) {
        for (int j = 0; j < HILBERT_GRID_DIM; j++) {
            for (int k = 0; k < HILBERT_GRID_DIM; k++) {

                bitmask_t hilbert_coords[3];
                hilbert_coords[0] = i;
                hilbert_coords[1] = j;
                hilbert_coords[2] = k;

                unsigned int bin = static_cast<unsigned int>(hilbert_c2i(3, HILBERT_N_BITS, hilbert_coords));
                bin_to_idx[i * HILBERT_GRID_DIM * HILBERT_GRID_DIM + j * HILBERT_GRID_DIM + k] = bin;
            }
        }
    }

    d_bin_to_idx_.copy_from(&bin_to_idx[0]);

    // estimate size needed to do radix sorting
    // reuse d_sort_keys_in_ rather than constructing a dummy output idxs buffer
    gpuErrchk(cub::DeviceRadixSort::SortPairs(
        nullptr,
        d_sort_storage_bytes_,
        d_sort_keys_in_.data,
        d_sort_keys_out_.data,
        d_sort_vals_in_.data,
        d_sort_keys_in_.data,
        N_));

    d_sort_storage_.realloc(d_sort_storage_bytes_);
}

HilbertSort::~HilbertSort() {};

void HilbertSort::sort_device(
    const int N,
    const unsigned int *d_atom_idxs,
    const double *d_coords,
    const double *d_box,
    unsigned int *d_output_perm,
    cudaStream_t stream) {
    if (N > N_) {
        throw std::runtime_error("number of idxs to sort must be less than or equal to N");
    }
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int B = ceil_divide(N, tpb);

    k_coords_to_kv_gather<<<B, tpb, 0, stream>>>(
        N, d_atom_idxs, d_coords, d_box, d_bin_to_idx_.data, d_sort_keys_in_.data, d_sort_vals_in_.data);

    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cub::DeviceRadixSort::SortPairs(
        d_sort_storage_.data,
        d_sort_storage_bytes_,
        d_sort_keys_in_.data,
        d_sort_keys_out_.data,
        d_sort_vals_in_.data,
        d_output_perm,
        N,
        0,                                 // begin bit
        sizeof(*d_sort_keys_in_.data) * 8, // end bit
        stream                             // cudaStream
        ));
}

std::vector<unsigned int> HilbertSort::sort_host(const int N, const double *h_coords, const double *h_box) {

    std::vector<unsigned int> h_atom_idxs(N);
    std::iota(h_atom_idxs.begin(), h_atom_idxs.end(), 0);

    DeviceBuffer<double> d_coords(N * 3);
    DeviceBuffer<double> d_box(3 * 3);
    DeviceBuffer<unsigned int> d_atom_idxs(h_atom_idxs);
    DeviceBuffer<unsigned int> d_perm(N);

    d_coords.copy_from(h_coords);
    d_box.copy_from(h_box);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->sort_device(N, d_atom_idxs.data, d_coords.data, d_box.data, d_perm.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    d_perm.copy_to(&h_atom_idxs[0]);
    return h_atom_idxs;
}

} // namespace timemachine
