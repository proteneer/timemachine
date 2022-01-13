#pragma once

#define FIXED_EXPONENT 0x1000000000

// we need to use a different level of precision for parameter derivatives
#define FIXED_EXPONENT_DU_DCHARGE 0x1000000000
#define FIXED_EXPONENT_DU_DSIG 0x2000000000
#define FIXED_EXPONENT_DU_DEPS 0x4000000000 // this is just getting silly

template <typename RealType, unsigned long long EXPONENT>
RealType __host__ __device__ __forceinline__ FIXED_TO_FLOAT_DU_DP(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v)) / EXPONENT;
}

template <typename RealType> RealType __host__ __device__ __forceinline__ FIXED_TO_FLOAT(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v)) / FIXED_EXPONENT;
}
