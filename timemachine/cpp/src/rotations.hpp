#pragma once

namespace timemachine {

template <typename RealType>
void rotate_coordinates_host(
    const int N, const int n_rotations, const double *coords, const RealType *quaternions, double *output);

} // namespace timemachine
