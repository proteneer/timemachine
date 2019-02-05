#pragma once
#include <array>
#include <complex>

namespace timemachine {

// Trick to allow type promotion below
// https://stackoverflow.com/questions/2647858/multiplying-complex-with-constant-in-c
template <typename T>
struct identity_t { typedef T type; };

/// Make working with std::complex<> nubmers suck less... allow promotion.
#define COMPLEX_OPS(OP)                                                 \
    template <typename _Tp>                                               \
    std::complex<_Tp>                                                     \
    operator OP(std::complex<_Tp> lhs, const typename identity_t<_Tp>::type & rhs) \
    {                                                                     \
    return lhs OP rhs;                                                  \
    }                                                                     \
    template <typename _Tp>                                               \
    std::complex<_Tp>                                                     \
    operator OP(const typename identity_t<_Tp>::type & lhs, const std::complex<_Tp> & rhs) \
    {                                                                     \
    return lhs OP rhs;                                                  \
    }
COMPLEX_OPS(+)
COMPLEX_OPS(-)
COMPLEX_OPS(*)
COMPLEX_OPS(/)
#undef COMPLEX_OPS


template<typename RealType, size_t N>
std::array<std::complex<RealType>, N> convert_to_complex(const std::array<RealType, N> &input) {
    std::array<std::complex<RealType>, N> result;
    for(size_t i=0; i < input.size(); i++) {
        result[i] = std::complex<RealType>(input[i], 0);
    }
    return result;
}

template<typename RealType>
std::vector<std::complex<RealType> > convert_to_complex(const std::vector<RealType> &input) {
    std::vector<std::complex<RealType> > result(input.size());
    for(size_t i=0; i < input.size(); i++) {
        result[i] = std::complex<RealType>(input[i], 0);
    }
    return result;
}

template<typename RealType>
RealType dot_product(
    RealType x0, RealType y0, RealType z0,
    RealType x1, RealType y1, RealType z1) {
    return x0*x1 + y0*y1 + z0*z1;
}

template<typename RealType>
void cross_product(
    RealType a1, RealType a2, RealType a3,
    RealType b1, RealType b2, RealType b3,
    RealType &s1, RealType &s2, RealType &s3) {

    s1 = a2*b3 - a3*b2;
    s2 = a3*b1 - a1*b3;
    s3 = a1*b2 - a2*b1;

}

template<typename RealType>
RealType norm(RealType x, RealType y, RealType z) {
    return sqrt(x*x + y*y + z*z);
}

}