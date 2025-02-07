#include "k_logsumexp.cuh"

namespace timemachine {

template void __global__ k_segmented_exp_sub_max<float>(const int, const int *, const float *, const float *, float *);
template void __global__
k_segmented_exp_sub_max<double>(const int, const int *, const double *, const double *, double *);

} // namespace timemachine
