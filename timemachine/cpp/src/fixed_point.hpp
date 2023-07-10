#pragma once

#define FIXED_EXPONENT 0x1000000000

// we need to use a different level of precision for parameter derivatives
#define FIXED_EXPONENT_DU_DCHARGE 0x1000000000
#define FIXED_EXPONENT_DU_DSIG 0x2000000000
#define FIXED_EXPONENT_DU_DEPS 0x4000000000 // this is just getting silly
#define FIXED_EXPONENT_DU_DW 0x1000000000

template <typename RealType, unsigned long long EXPONENT>
RealType __host__ __device__ __forceinline__ FIXED_TO_FLOAT_DU_DP(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v)) / EXPONENT;
}

template <typename RealType> RealType __host__ __device__ __forceinline__ FIXED_TO_FLOAT(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v)) / FIXED_EXPONENT;
}

template <typename RealType> bool inline __device__ is_overflow(RealType u) {
    return u > nextafter(static_cast<RealType>(LLONG_MAX), static_cast<RealType>(0.0)) ||
           u < nextafter(static_cast<RealType>(LLONG_MIN), static_cast<RealType>(0.0));
}

template <typename RealType> bool inline __device__ energy_overflowed(RealType u_orig, int &overflow_count) {
    RealType u = u_orig * FIXED_EXPONENT;
    if (!isfinite(u) || is_overflow(u)) {
        overflow_count++;
        return true;
    }
    return false;
}
