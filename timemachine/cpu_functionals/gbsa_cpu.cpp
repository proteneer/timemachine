#include <vector>
#include <iostream>
#include <cmath>
#include "utils.hpp"

std::vector<double> r_i_real(5, 0);
std::vector<bool> r_i_flag(5, 0);

std::vector<std::complex<double> > r_i_complex(5, 0);
std::vector<bool> r_i_flag_complex(5, 0);

template<typename NumericType>
NumericType radii(NumericType x_i, std::vector<NumericType> xs) {
    NumericType sum = 0;
    for(auto j=0; j < xs.size(); j++) {
        NumericType dx = x_i - xs[j];
        sum += dx*dx;
    }
    return sum;
}

template<typename NumericType>
NumericType energy(
    size_t idx,
    size_t jdx,
    std::vector<NumericType> all_xs) {
    
    if(!r_i_flag[idx]) {
        r_i_real[idx] = radii(all_xs[idx], all_xs);
    }
    if(!r_i_flag[jdx]) {
        r_i_real[jdx] = radii(all_xs[jdx], all_xs);
    }
    NumericType r_i = r_i_real[idx];
    NumericType r_j = r_i_real[jdx];

    NumericType dx = r_i - r_j;
    NumericType dij = sqrt(dx*dx);

    return r_i * r_j * dij;
}

template<typename NumericType>
NumericType energy_complex(
    size_t idx,
    size_t jdx,
    std::vector<NumericType> all_xs) {
    
    if(!r_i_flag_complex[idx]) {
        r_i_complex[idx] = radii(all_xs[idx], all_xs);
    }
    if(!r_i_flag_complex[jdx]) {
        r_i_complex[jdx] = radii(all_xs[jdx], all_xs);
    }
    NumericType r_i = r_i_complex[idx];
    NumericType r_j = r_i_complex[jdx];

    NumericType dx = r_i - r_j;
    NumericType dij = sqrt(dx*dx);

    return r_i * r_j * dij;
}

template<typename NumericType>
NumericType outer_loop(std::vector<NumericType> xs) {
    NumericType nrg_sum = 0;
    std::vector<std::complex<NumericType> > cxs = timemachine::convert_to_complex<NumericType>(xs);
    NumericType step = 1e-100;
    for(size_t i=0; i < xs.size(); i++) {
        for(size_t j=0; j < xs.size(); j++) {
            if(i == j) {
                continue;
            }
            
            // energy
            nrg_sum += energy(i, j, xs);

            // dE/dxi
            cxs[i] = std::complex<NumericType>(xs[i], step);
            std::complex<NumericType> dE_dxi = energy_complex(i, j, cxs);

            std::cout << dE_dxi.imag()/step << std::endl;
            // dE/dxj




        }
    }
    return nrg_sum;
}


int main() {
    std::vector<double> input({1.0, 2.0, 5.5, 1.0, 5.0});
    std::cout << outer_loop(input) << std::endl;
}