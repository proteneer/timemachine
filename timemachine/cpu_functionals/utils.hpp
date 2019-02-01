#pragma once
#include <array>
#include <complex>

namespace timemachine {

template<typename RealType, size_t N>
std::array<std::complex<RealType>, N> convert_to_complex(const std::array<RealType, N> &input) {
    std::array<std::complex<RealType>, N> result;
    for(size_t i=0; i < input.size(); i++) {
        result[i] = std::complex<RealType>(input[i], 0);
    }
    return result;
}

}