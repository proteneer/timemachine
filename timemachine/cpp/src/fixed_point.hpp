#pragma once

#define FIXED_EXPONENT             0x1000000000

template<typename RealType>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED(RealType v) {
    return static_cast<unsigned long long>((long long)(v*FIXED_EXPONENT));
}

template<typename RealType>
RealType __device__ __forceinline__ FIXED_TO_FLOAT(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v))/FIXED_EXPONENT;
}