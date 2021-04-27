#pragma once

// cuda specific version
#include "../fixed_point.hpp"

// we need to use a different level of precision for parameter derivatives
#define FIXED_EXPONENT_DU_DCHARGE 0x1000000000
#define FIXED_EXPONENT_DU_DSIG    0x2000000000
#define FIXED_EXPONENT_DU_DEPS    0x4000000000 // this is just getting silly


template<typename RealType, unsigned long long EXPONENT>
RealType __device__ __forceinline__ FIXED_TO_FLOAT_DU_DP(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v))/EXPONENT;
}

// (ytz): courtesy of @scottlegrand/NVIDIA, even faster conversion
// This was original a hack to improve perf on Maxwell, that is now needed for Ampere
long long __device__ __forceinline__ real_to_int64(float x) {
#if __CUDA_ARCH__ == 860
  float z = x * (float)0x1.00000p-32;
  int hi = __float2int_rz( z );                         // First convert high bits
  float delta = x - ((float)0x1.00000p32*((float)hi));  // Check remainder sign
  int test = (__float_as_uint(delta) > 0xbf000000);
  int lo = __float2uint_rn(fabsf(delta));               // Convert the (unsigned) remainder
  lo = (test) ? -lo: lo;
  hi -= test;                                           // Two's complement correction
  long long res = __double_as_longlong(__hiloint2double(hi,lo)); // Return 64-bit result
  return res;
#else
  return llrintf(x);
#endif
}

// (ytz): reference version, left here for pedagogical reasons, do not remove.
// long long __device__ __forceinline__ real_to_int64(float x) {
//     if(x == 0) {
//         return 0;
//     }
//     unsigned int * ptr = reinterpret_cast<unsigned int *>(&x);
//     // get the sign bit
//     unsigned int s = *ptr >> 31; // (TBD) there's a faster alternative to shifting here
//     // get the exponent
//     unsigned int e = *ptr & 0x7f800000;
//     // shift the exponent and apply the exponent bias
//     e >>= 23;
//     e -= 127;
//     // get the mantissa and append a 1 at the front
//     long long m = *ptr & 0x007fffff;
//     m |= 1 << 23;
//     // 24 - e, either left shift or right shift depending on exponent
//     int shift = 23 - e;
//     m = shift > 0 ? m >> shift : m << -shift;
//     // copy sign
//     if(s == 1) {
//         m = -m;
//     }

//     return m;
// }

long long __device__ __forceinline__ real_to_int64(double x) {
    return llrint(x);
}

// generic version
template<typename RealType>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v*FIXED_EXPONENT));
}

template<typename RealType, unsigned long long EXPONENT>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_DU_DP(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v*EXPONENT));
}

template<typename RealType>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_NONBONDED(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v*FIXED_EXPONENT));
}

template<typename RealType>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_BONDED(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v*FIXED_EXPONENT));
}

