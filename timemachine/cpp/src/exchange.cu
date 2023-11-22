#include "device_buffer.hpp"
#include "exchange.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_exchange.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"

namespace timemachine {

template <typename RealType>
std::array<std::vector<int>, 2> get_inner_and_outer_mols(
    const std::vector<int> &center_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &box,
    const std::vector<std::vector<int>> &group_idxs,
    const RealType radius) {

    const int num_mols = group_idxs.size();
    DeviceBuffer<int> d_center_atoms(center_atoms);
    DeviceBuffer<double> d_coords(coords);
    DeviceBuffer<double> d_box(box);

    DeviceBuffer<int> d_flags(num_mols);

    DeviceBuffer<int> d_mol_offsets(get_mol_offsets(group_idxs));
    DeviceBuffer<int> d_atom_indices(get_atom_indices(group_idxs));

    DeviceBuffer<RealType> d_centroid(3);
    cudaStream_t stream = static_cast<cudaStream_t>(0);

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    k_compute_centroid_of_atoms<<<1, tpb, 0, stream>>>(
        static_cast<int>(d_center_atoms.length), d_center_atoms.data, d_coords.data, d_centroid.data);
    gpuErrchk(cudaPeekAtLastError());

    const RealType square_radius = radius * radius;

    k_flag_mols_inner_outer<<<ceil_divide(num_mols, tpb), tpb, 0, stream>>>(
        num_mols,
        d_atom_indices.data,
        d_mol_offsets.data,
        d_centroid.data,
        square_radius,
        d_coords.data,
        d_box.data,
        d_flags.data);
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<int> h_inner_idxs(0);
    std::vector<int> h_outer_idxs(0);
    std::vector<int> h_flags(d_flags.length);
    d_flags.copy_to(&h_flags[0]);
    for (int i = 0; i < h_flags.size(); i++) {
        if (h_flags[i] == 1) {
            h_inner_idxs.push_back(i);
        } else {
            h_outer_idxs.push_back(i);
        }
    }

    return std::array<std::vector<int>, 2>({h_inner_idxs, h_outer_idxs});
}

template std::array<std::vector<int>, 2> get_inner_and_outer_mols<float>(
    const std::vector<int> &center_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &box,
    const std::vector<std::vector<int>> &group_idxs,
    const float radius);
template std::array<std::vector<int>, 2> get_inner_and_outer_mols<double>(
    const std::vector<int> &center_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &box,
    const std::vector<std::vector<int>> &group_idxs,
    const double radius);

} // namespace timemachine
