#pragma once

namespace timemachine {

template <typename RealType>
void rotate_coordinates_host(
    const int N, const int n_rotations, const double *coords, const RealType *quaternions, double *output);

template <typename RealType>
void rotate_coordinates_and_translate_mol_host(
    const int N,
    const int batch_size,
    const double *mol_coords,
    const double *box,
    const RealType *quaternion,
    const RealType *translation,
    double *output);

} // namespace timemachine
