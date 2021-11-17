#pragma once

template <typename RealType, int D>
void fast_nonbonded_normal(
    const RealType *coords,
    const RealType *coords_tangents,
    const RealType *params,
    const int *param_idxs,
    const double cutoff,
    int N,
    int P,
    RealType *out_coords,
    RealType *out_coords_tangents,
    RealType *out_params_tangents);
