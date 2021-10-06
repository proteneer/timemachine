
namespace timemachine {
/*
Optimally align x2 onto x1.
*/
void rmsd_align_cpu(
    const int N,
    const double *x1_raw,
    const double *x2_raw,
    double *x2_aligned_raw);

}