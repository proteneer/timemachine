#pragma once
#include "cuda_runtime.h"
#include "device_buffer.hpp"
#include "math_utils.cuh"
#include "types.hpp"
#include <memory>
#include <numeric>
#include <vector>

namespace timemachine {

class HilbertSort {

private:
    const int N_;
    // used for hilbert sorting
    DeviceBuffer<unsigned int>
        d_bin_to_idx_; // mapping from HILBERT_GRID_DIMxHILBERT_GRID_DIMxHILBERT_GRID_DIM grid to hilbert curve index
    DeviceBuffer<unsigned int> d_sort_keys_in_;
    DeviceBuffer<unsigned int> d_sort_keys_out_;
    DeviceBuffer<unsigned int> d_sort_vals_in_;
    std::unique_ptr<DeviceBuffer<char>> d_sort_storage_;
    size_t d_sort_storage_bytes_;

public:
    // N - number of atoms
    HilbertSort(const int N);

    ~HilbertSort();

    void sort_device(
        const int N,
        const unsigned int *d_atom_idxs,
        const CoordsType *d_coords,
        const CoordsType *d_box,
        unsigned int *d_output_perm,
        cudaStream_t stream);

    std::vector<unsigned int> sort_host(const int N, const CoordsType *h_coords, const CoordsType *h_box);
};

} // namespace timemachine
