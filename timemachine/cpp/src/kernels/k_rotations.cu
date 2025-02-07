#include "k_rotations.cuh"

namespace timemachine {

template void __global__ k_rotate_coordinates<float>(int, int, const double *, const float *, double *);
template void __global__ k_rotate_coordinates<double>(int, int, const double *, const double *, double *);

template void __global__ k_rotate_and_translate_mols<float, false>(
    const int,
    const int,
    const int *,
    const double *,
    const double *,
    const int *,
    const int *,
    const float *,
    const float *,
    double *);
template void __global__ k_rotate_and_translate_mols<float, true>(
    const int,
    const int,
    const int *,
    const double *,
    const double *,
    const int *,
    const int *,
    const float *,
    const float *,
    double *);
template void __global__ k_rotate_and_translate_mols<double, false>(
    const int,
    const int,
    const int *,
    const double *,
    const double *,
    const int *,
    const int *,
    const double *,
    const double *,
    double *);
template void __global__ k_rotate_and_translate_mols<double, true>(
    const int,
    const int,
    const int *,
    const double *,
    const double *,
    const int *,
    const int *,
    const double *,
    const double *,
    double *);

} // namespace timemachine
