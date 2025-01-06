#pragma once

// cuda specific version
#include "../fixed_point.hpp"

namespace timemachine {

// (ytz): courtesy of @scottlegrand/NVIDIA, even faster conversion
// This was original a hack to improve perf on Maxwell, that is now needed since Ampere
long long __device__ __forceinline__ real_to_int64(float x) {
#if __CUDA_ARCH__ >= 610
    float z = x * (float)0x1.00000p-32;
    int hi = __float2int_rz(z);                                     // First convert high bits
    float delta = x - ((float)0x1.00000p32 * ((float)hi));          // Check remainder sign
    int test = (__float_as_uint(delta) > 0xbf000000);
    int lo = __float2uint_rn(fabsf(delta));                         // Convert the (unsigned) remainder
    lo = (test) ? -lo : lo;
    hi -= test;                                                     // Two's complement correction
    long long res = __double_as_longlong(__hiloint2double(hi, lo)); // Return 64-bit result
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

long long __device__ __forceinline__ real_to_int64(double x) { return llrint(x); }

// generic version
template <typename RealType> unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v * FIXED_EXPONENT));
}

template <typename RealType, unsigned long long EXPONENT>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_DU_DP(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v * EXPONENT));
}

template <typename RealType> unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_NONBONDED(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v * FIXED_EXPONENT));
}

template <typename RealType> unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_BONDED(RealType v) {
    return static_cast<unsigned long long>(real_to_int64(v * FIXED_EXPONENT));
}

/* FLOAT_TO_FIXED_ENERGY converts floating point energies into fixed point. Values beyond LLONG_MAX/LLONG_MIN
* (or non-finite) will be capped to LLONG_MAX. This is accumulated into __int128 which handles the positive/negative overflows,
* allows to account for overflows triggered by the summation of energies.
*
* The energy values are only considered valid between the values LLONG_MIN and LLONG_MAX, and we use __int128 to be able to detect that the energies are invalid.
* If there are individual interactions that are overflows (beyond limit or non-finite) we set LLONG_MAX to be the value. This way
* the energy is beyond the bounds and only exclusion cancellations (only done between NonbondedAllPairs/IxnGroups and NonbondedPairList<..., true>) do not trigger invalid energies.
* In the case where all interactions are within the bounds, we can still overflows due to summation which int128 allows us to detect.
*
* Example of Summation overflow
* -----------------------------
* accumulated_energy = sum([LLONG_MAX - 1, LLONG_MAX - 1])
* (__int128)accumulated_energy > LLONG_MAX  - Correctly detects that energies are beyond valid range
* (long long)accumulated_energy > LLONG_MAX - Overflows and results in seemingly valid energies
*/
template <typename RealType> __int128 __device__ __forceinline__ FLOAT_TO_FIXED_ENERGY(RealType u_orig) {
    RealType u = u_orig * FIXED_EXPONENT;
    // All clashes (beyond representation of long long) are treated as LLONG_MAX, to avoid clashes of different signs but non-identical values
    // cancelling out.
    if (!isfinite(u) || static_cast<__int128>(u) >= static_cast<__int128>(LLONG_MAX) ||
        static_cast<__int128>(u) <= static_cast<__int128>(LLONG_MIN)) {
        return static_cast<__int128>(LLONG_MAX);
    } else {
        return static_cast<__int128>(real_to_int64(u));
    }
}

} // namespace timemachine
