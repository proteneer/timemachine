#pragma once

#define FIXED_EXPONENT 0x1000000000

template <typename RealType> RealType __host__ __device__ __forceinline__ FIXED_TO_FLOAT(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v)) / FIXED_EXPONENT;
}
