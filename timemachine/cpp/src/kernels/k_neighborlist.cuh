#pragma once


void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const double *coords,
    double *block_bounds_ctr,
    double *block_bounds_ext);