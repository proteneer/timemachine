#include <algorithm>
#include <vector>

#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_neighborlist.cuh"
#include "kernels/k_nonbonded_common.cuh"
#include "neighborlist.hpp"
#include "set_utils.hpp"

namespace timemachine {

template <typename RealType> Neighborlist<RealType>::Neighborlist(const int N) : max_size_(N), N_(N), NC_(N), NR_(N) {
    if (N == 0) {
        throw std::runtime_error("Neighborlist N must be at least 1");
    }
    const int column_blocks = this->num_column_blocks();
    const int row_blocks = this->num_row_blocks();
    const int Y = this->Y();

    const unsigned long long MAX_TILE_BUFFER = row_blocks * column_blocks;
    const unsigned long long MAX_ATOM_BUFFER = this->max_ixn_count();

    // interaction buffers
    cudaSafeMalloc(&d_ixn_count_, 1 * sizeof(*d_ixn_count_));
    cudaSafeMalloc(&d_ixn_tiles_, MAX_TILE_BUFFER * sizeof(*d_ixn_tiles_));
    cudaSafeMalloc(&d_ixn_atoms_, MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_));
    cudaSafeMalloc(&d_trim_atoms_, column_blocks * Y * TILE_SIZE * sizeof(*d_trim_atoms_));

    // bounding box buffers
    cudaSafeMalloc(&d_row_block_bounds_ctr_, row_blocks * 3 * sizeof(*d_row_block_bounds_ctr_));
    cudaSafeMalloc(&d_row_block_bounds_ext_, row_blocks * 3 * sizeof(*d_row_block_bounds_ext_));
    cudaSafeMalloc(&d_column_block_bounds_ctr_, column_blocks * 3 * sizeof(*d_column_block_bounds_ctr_));
    cudaSafeMalloc(&d_column_block_bounds_ext_, column_blocks * 3 * sizeof(*d_column_block_bounds_ext_));

    // Row and column indice arrays
    cudaSafeMalloc(&d_column_idxs_, max_size_ * sizeof(*d_column_idxs_));
    cudaSafeMalloc(&d_row_idxs_, max_size_ * sizeof(*d_row_idxs_));

    this->reset_row_idxs();
}

template <typename RealType> Neighborlist<RealType>::~Neighborlist() {
    gpuErrchk(cudaFree(d_column_idxs_));
    gpuErrchk(cudaFree(d_row_idxs_));

    gpuErrchk(cudaFree(d_ixn_count_));
    gpuErrchk(cudaFree(d_ixn_tiles_));
    gpuErrchk(cudaFree(d_ixn_atoms_));
    gpuErrchk(cudaFree(d_trim_atoms_));

    gpuErrchk(cudaFree(d_row_block_bounds_ctr_));
    gpuErrchk(cudaFree(d_row_block_bounds_ext_));
    gpuErrchk(cudaFree(d_column_block_bounds_ctr_));
    gpuErrchk(cudaFree(d_column_block_bounds_ext_));
}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_host(
    const int N, const double *h_coords, const double *h_box, double *h_bb_ctrs, double *h_bb_exts) {

    const int D = 3;
    DeviceBuffer<double> d_coords(N * D);
    DeviceBuffer<double> d_box(D * D);

    std::vector<RealType> h_block_bounds_centers(this->num_column_blocks() * 3);
    std::vector<RealType> h_block_bounds_extents(this->num_column_blocks() * 3);

    d_coords.copy_from(h_coords);
    d_box.copy_from(h_box);

    this->compute_block_bounds_device(N, D, d_coords.data, d_box.data, static_cast<cudaStream_t>(0));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(
        &h_block_bounds_centers[0],
        d_column_block_bounds_ctr_,
        this->num_column_blocks() * 3 * sizeof(*d_column_block_bounds_ctr_),
        cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(
        &h_block_bounds_extents[0],
        d_column_block_bounds_ext_,
        this->num_column_blocks() * 3 * sizeof(*d_column_block_bounds_ext_),
        cudaMemcpyDeviceToHost));

    // Handle the float -> double, doing a direct copy from a double buffer to a float buffer results in garbage values
    for (auto i = 0; i < h_block_bounds_centers.size(); i++) {
        h_bb_ctrs[i] = h_block_bounds_centers[i];
    }
    for (auto i = 0; i < h_block_bounds_extents.size(); i++) {
        h_bb_exts[i] = h_block_bounds_extents[i];
    }
}

// Return the number of tiles that interact
template <typename RealType> unsigned int Neighborlist<RealType>::num_tile_ixns() {
    unsigned int h_ixn_count;
    gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count_, 1 * sizeof(*d_ixn_count_), cudaMemcpyDeviceToHost));
    return h_ixn_count;
}

template <typename RealType>
std::vector<std::vector<int>>
Neighborlist<RealType>::get_nblist_host(int N, const double *h_coords, const double *h_box, const double cutoff) {

    if (N != N_) {
        throw std::runtime_error("N != N_");
    }

    DeviceBuffer<double> d_coords(N * 3);
    DeviceBuffer<double> d_box(3 * 3);
    d_coords.copy_from(h_coords);
    d_box.copy_from(h_box);

    this->build_nblist_device(N, d_coords.data, d_box.data, cutoff, static_cast<cudaStream_t>(0));

    gpuErrchk(cudaDeviceSynchronize());
    const int column_blocks = this->num_column_blocks();
    const int row_blocks = this->num_row_blocks();

    unsigned long long MAX_TILE_BUFFER = row_blocks * column_blocks;
    unsigned long long MAX_ATOM_BUFFER = this->max_ixn_count();

    unsigned int h_ixn_count;
    gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count_, 1 * sizeof(*d_ixn_count_), cudaMemcpyDeviceToHost));
    std::vector<int> h_ixn_tiles(MAX_TILE_BUFFER);
    std::vector<unsigned int> h_ixn_atoms(MAX_ATOM_BUFFER);
    gpuErrchk(
        cudaMemcpy(&h_ixn_tiles[0], d_ixn_tiles_, MAX_TILE_BUFFER * sizeof(*d_ixn_tiles_), cudaMemcpyDeviceToHost));
    gpuErrchk(
        cudaMemcpy(&h_ixn_atoms[0], d_ixn_atoms_, MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> ixn_list(row_blocks, std::vector<int>());
    for (int i = 0; i < h_ixn_count; i++) {
        int tile_idx = h_ixn_tiles[i];
        for (int j = 0; j < TILE_SIZE; j++) {
            int atom_j_idx = h_ixn_atoms[i * TILE_SIZE + j];
            if (atom_j_idx < N) {
                ixn_list[tile_idx].push_back(atom_j_idx);
            }
        }
    }

    return ixn_list;
}

template <typename RealType>
void Neighborlist<RealType>::build_nblist_device(
    const int N, const double *d_coords, const double *d_box, const double cutoff, const cudaStream_t stream) {

    const int D = 3;
    this->compute_block_bounds_device(N, D, d_coords, d_box, stream);
    const int tpb = TILE_SIZE;
    const int row_blocks = this->num_row_blocks();
    const int Y = this->Y();

    dim3 dimGrid(row_blocks, Y, 1); // block x, y, z dims

    // (ytz): TBD shared memory, stream
    if (this->compute_upper_triangular()) {
        // Compute only the upper triangle as rows and cols are the same
        // pass duplicates of column coords and the bounding boxes
        k_find_blocks_with_ixns<RealType, true><<<dimGrid, tpb, 0, stream>>>(
            N_,
            NC_,
            NR_,
            d_column_idxs_,
            d_row_idxs_,
            d_column_block_bounds_ctr_,
            d_column_block_bounds_ext_,
            d_column_block_bounds_ctr_,
            d_column_block_bounds_ext_,
            d_coords,
            d_box,
            d_ixn_count_,
            d_ixn_tiles_,
            d_ixn_atoms_,
            d_trim_atoms_,
            cutoff);
    } else {
        k_find_blocks_with_ixns<RealType, false><<<dimGrid, tpb, 0, stream>>>(
            N_,
            NC_,
            NR_,
            d_column_idxs_,
            d_row_idxs_,
            d_column_block_bounds_ctr_,
            d_column_block_bounds_ext_,
            d_row_block_bounds_ctr_,
            d_row_block_bounds_ext_,
            d_coords,
            d_box,
            d_ixn_count_,
            d_ixn_tiles_,
            d_ixn_atoms_,
            d_trim_atoms_,
            cutoff);
    }

    gpuErrchk(cudaPeekAtLastError());
    k_compact_trim_atoms<<<row_blocks, tpb, 0, stream>>>(
        N_, Y, d_trim_atoms_, d_ixn_count_, d_ixn_tiles_, d_ixn_atoms_);

    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Neighborlist<RealType>::maybe_build_nblist_device(
    const int N,
    const double *d_coords,
    const double *d_box,
    const double cutoff,
    const cudaStream_t stream,
    int *d_rebuild_nblist,
    double *d_nblist_x,
    double *d_nblist_box) {

    const int D = 3;
    this->maybe_compute_block_bounds_device(N, D, d_coords, d_box, stream, d_rebuild_nblist);
    const int tpb = TILE_SIZE;
    const int row_blocks = this->num_row_blocks();
    const int Y = this->Y();

    dim3 dimGrid(row_blocks, Y, 1); // block x, y, z dims

    // (ytz): TBD shared memory, stream
    if (this->compute_upper_triangular()) {
        // Compute only the upper triangle as rows and cols are the same
        // pass duplicates of column coords and the bounding boxes
        k_maybe_find_blocks_with_ixns<RealType, true><<<1, 1, 0, stream>>>(
            N_,
            NC_,
            NR_,
            d_column_idxs_,
            d_row_idxs_,
            d_column_block_bounds_ctr_,
            d_column_block_bounds_ext_,
            d_column_block_bounds_ctr_,
            d_column_block_bounds_ext_,
            d_coords,
            d_box,
            d_ixn_count_,
            d_ixn_tiles_,
            d_ixn_atoms_,
            d_trim_atoms_,
            cutoff,
            d_rebuild_nblist,
            dimGrid,
            tpb);
    } else {
        k_maybe_find_blocks_with_ixns<RealType, false><<<1, 1, 0, stream>>>(
            N_,
            NC_,
            NR_,
            d_column_idxs_,
            d_row_idxs_,
            d_column_block_bounds_ctr_,
            d_column_block_bounds_ext_,
            d_row_block_bounds_ctr_,
            d_row_block_bounds_ext_,
            d_coords,
            d_box,
            d_ixn_count_,
            d_ixn_tiles_,
            d_ixn_atoms_,
            d_trim_atoms_,
            cutoff,
            d_rebuild_nblist,
            dimGrid,
            tpb);
    }

    gpuErrchk(cudaPeekAtLastError());
    k_maybe_compact_trim_atoms<<<row_blocks, tpb, 0, stream>>>(
        N_, Y, d_trim_atoms_, d_ixn_count_, d_ixn_tiles_, d_ixn_atoms_, d_rebuild_nblist, row_blocks, tpb);

    k_maybe_reset_nblist_rebuild<RealType><<<1, 1, 0, stream>>>(N, tpb, d_rebuild_nblist, d_nblist_x, d_nblist_box);

    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_device(
    const int N,            // Number of atoms
    const int D,            // Box dimensions
    const double *d_coords, // [N*3]
    const double *d_box,    // [D*3]
    const cudaStream_t stream) {

    if (D != 3) {
        throw std::runtime_error("D != 3");
    }

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    k_find_block_bounds<RealType><<<ceil_divide(NC_, tpb), tpb, 0, stream>>>(
        this->num_column_blocks(),
        NC_,
        d_column_idxs_,
        d_coords,
        d_box,
        d_column_block_bounds_ctr_,
        d_column_block_bounds_ext_,
        d_ixn_count_);
    gpuErrchk(cudaPeekAtLastError());
    // In the case of upper triangle of the matrix, the column and row indices are the same, so only compute block ixns for both
    // when they are different
    if (!this->compute_upper_triangular()) {
        k_find_block_bounds<RealType><<<ceil_divide(NR_, tpb), tpb, 0, stream>>>(
            this->num_row_blocks(),
            NR_,
            d_row_idxs_,
            d_coords,
            d_box,
            d_row_block_bounds_ctr_,
            d_row_block_bounds_ext_,
            d_ixn_count_);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template <typename RealType>
void Neighborlist<RealType>::maybe_compute_block_bounds_device(
    const int N,            // Number of atoms
    const int D,            // Box dimensions
    const double *d_coords, // [N*3]
    const double *d_box,    // [D*3]
    const cudaStream_t stream,
    const int *d_rebuild_nblist) {

    if (D != 3) {
        throw std::runtime_error("D != 3");
    }

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    k_maybe_find_block_bounds<RealType><<<1, 1, 0, stream>>>(
        this->num_column_blocks(),
        NC_,
        d_column_idxs_,
        d_coords,
        d_box,
        d_column_block_bounds_ctr_,
        d_column_block_bounds_ext_,
        d_ixn_count_,
        d_rebuild_nblist,
        tpb);
    gpuErrchk(cudaPeekAtLastError());
    // In the case of upper triangle of the matrix, the column and row indices are the same, so only compute block ixns for both
    // when they are different
    if (!this->compute_upper_triangular()) {
        k_maybe_find_block_bounds<RealType><<<1, 1, 0, stream>>>(
            this->num_row_blocks(),
            NR_,
            d_row_idxs_,
            d_coords,
            d_box,
            d_row_block_bounds_ctr_,
            d_row_block_bounds_ext_,
            d_ixn_count_,
            d_rebuild_nblist,
            tpb);
        gpuErrchk(cudaPeekAtLastError());
    }
};
template <typename RealType> void Neighborlist<RealType>::set_row_idxs(std::vector<unsigned int> row_idxs) {
    if (row_idxs.size() == 0) {
        throw std::runtime_error("idxs can't be empty");
    }
    std::set<unsigned int> unique_idxs(row_idxs.begin(), row_idxs.end());
    if (unique_idxs.size() != row_idxs.size()) {
        throw std::runtime_error("atom indices must be unique");
    }
    if (row_idxs.size() >= N_) {
        throw std::runtime_error("number of idxs must be less than N");
    }
    if (*std::max_element(row_idxs.begin(), row_idxs.end()) >= N_) {
        throw std::runtime_error("indices values must be less than N");
    }
    std::vector<unsigned int> column_indices = get_indices_difference<unsigned int>(N_, unique_idxs);
    const size_t row_count = row_idxs.size();
    const size_t col_count = column_indices.size();

    DeviceBuffer<unsigned int> row_idx_buffer(row_count);
    DeviceBuffer<unsigned int> column_idx_buffer(col_count);

    row_idx_buffer.copy_from(&row_idxs[0]);

    column_idx_buffer.copy_from(&column_indices[0]);

    this->set_idxs_device(
        col_count, row_count, column_idx_buffer.data, row_idx_buffer.data, static_cast<cudaStream_t>(0));
    gpuErrchk(cudaDeviceSynchronize());
}

template <typename RealType> void Neighborlist<RealType>::reset_row_idxs() {
    const cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->reset_row_idxs_device(stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

template <typename RealType> void Neighborlist<RealType>::reset_row_idxs_device(const cudaStream_t stream) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(N_, tpb);
    // Fill the indices with the 0 to N-1 indices, indicating 'normal' neighborlist operation
    k_arange<<<blocks, tpb, 0, stream>>>(N_, d_column_idxs_);
    gpuErrchk(cudaPeekAtLastError());
    k_arange<<<blocks, tpb, 0, stream>>>(N_, d_row_idxs_);
    gpuErrchk(cudaPeekAtLastError());
    this->NR_ = N_;
    this->NC_ = N_;
}

template <typename RealType> void Neighborlist<RealType>::resize(const int size) {
    const cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->resize_device(size, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

// Resize the Neighborlist to function on a different size. Note that this only allows
// finding interactions on a smaller set of the system, will not increase the size of the underlying buffers.
template <typename RealType> void Neighborlist<RealType>::resize_device(const int size, const cudaStream_t stream) {
    if (size <= 0) {
        throw std::runtime_error("size is must be at least 1");
    }
    if (size > max_size_) {
        throw std::runtime_error(
            "size is greater than max size: " + std::to_string(size) + " > " + std::to_string(max_size_));
    }
    this->N_ = size;
    this->reset_row_idxs_device(stream);
}

// set_idxs_device is for use when idxs exist on the GPU already and are used as the new idxs to compute the neighborlist on.
template <typename RealType>
void Neighborlist<RealType>::set_idxs_device(
    const int NC,
    const int NR,
    unsigned int *d_in_column_idxs,
    unsigned int *d_in_row_idxs,
    const cudaStream_t stream) {
    if (NC + NR != N_) {
        throw std::runtime_error("Total of indices must equal N");
    }
    if (NC == 0 || NR == 0) {
        throw std::runtime_error("Number of column and row indices must be non-zero");
    }
    const size_t tpb = DEFAULT_THREADS_PER_BLOCK;

    // The indices must already be on the GPU and are copied into the neighborlist buffers.
    gpuErrchk(cudaMemcpyAsync(
        d_column_idxs_, d_in_column_idxs, NC * sizeof(*d_column_idxs_), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_row_idxs_, d_in_row_idxs, NR * sizeof(*d_row_idxs_), cudaMemcpyDeviceToDevice, stream));

    // Fill in the rest of values with N, potentially redundant
    k_initialize_array<unsigned int><<<ceil_divide(NR, tpb), tpb, 0, stream>>>(NR, d_column_idxs_ + NC, N_);
    gpuErrchk(cudaPeekAtLastError());
    k_initialize_array<unsigned int><<<ceil_divide(NC, tpb), tpb, 0, stream>>>(NC, d_row_idxs_ + NR, N_);
    gpuErrchk(cudaPeekAtLastError());

    // Update the row and column counts
    this->NR_ = NR;
    this->NC_ = NC;

    const unsigned long long MAX_ATOM_BUFFER = this->max_ixn_count();
    // Clear the atom ixns, to avoid reuse
    // Set to max value, ie greater than N. Note that Memset is on bytes, which is why it is UCHAR_MAX
    gpuErrchk(cudaMemsetAsync(d_ixn_atoms_, UCHAR_MAX, MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_), stream));
}

template <typename RealType> bool Neighborlist<RealType>::compute_upper_triangular() const {
    return NR_ == N_ && NC_ == N_;
};

template <typename RealType> int Neighborlist<RealType>::num_column_blocks() const {
    return ceil_divide(NC_, TILE_SIZE);
};

template <typename RealType> int Neighborlist<RealType>::Y() const {
    return ceil_divide(this->num_column_blocks(), WARP_SIZE);
};

template <typename RealType> int Neighborlist<RealType>::num_row_blocks() const { return ceil_divide(NR_, TILE_SIZE); }

// max_ixn_count determines the number of tile-atom interaction counts. For each tile that interacts with another
// it can have TILE_SIZE tile-atom interactions. Note that d_ixn_count_ is only the number of tile-tile interactions, and differs by a factor of TILE_SIZE
template <typename RealType> int Neighborlist<RealType>::max_ixn_count() const {
    // The maximum number of tile-atom interactions is that of all tile-tile interactions multiplied by TILE_SIZE.
    // Use the maximum value of N to compute the size of the upper triangular matrix to support any set of row indices.
    const int n_blocks = ceil_divide(max_size_, TILE_SIZE);
    int max_tile_tile_interactions = (n_blocks * (n_blocks + 1)) / 2;
    // Each tile-tile interaction can have TILE_SIZE tile-atom interactions
    return max_tile_tile_interactions * TILE_SIZE;
}

template class Neighborlist<double>;
template class Neighborlist<float>;

} // namespace timemachine
