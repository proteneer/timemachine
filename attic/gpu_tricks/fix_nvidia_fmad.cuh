// These are two lines of code are to deal with the formation of a non-commutative fma.
// For more information, see: https://github.com/proteneer/timemachine/issues/386
float __device__ __forceinline__ fix_nvidia_fmad(float a, float b, float c, float d) {
    return __fmul_rn(a, b) + __fmul_rn(c, d);
}

double __device__ __forceinline__ fix_nvidia_fmad(double a, double b, double c, double d) {
    return __dmul_rn(a, b) + __dmul_rn(c, d);
}
