#include "types.hpp"

namespace timemachine {
/*
Optimally align x2 onto x1.
*/
void rmsd_align_cpu(const int N, const CoordsType *x1_raw, const CoordsType *x2_raw, CoordsType *x2_aligned_raw);

} // namespace timemachine
