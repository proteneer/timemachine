#pragma once

#define FIXED_EXPONENT             0x1000000000
#define FIXED_BORN_PSI             0x40000000000
#define FIXED_EXPONENT_BORN_FORCES 0x100000000

template<typename RealType>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED(RealType v) {
    return static_cast<unsigned long long>((long long)(v*FIXED_EXPONENT));
}

template<typename RealType>
RealType __device__ __forceinline__ FIXED_TO_FLOAT(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v))/FIXED_EXPONENT;
}