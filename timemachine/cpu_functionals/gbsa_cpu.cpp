#include <vector>
#include <iostream>
#include <set>
#include <cmath>
#include "utils.hpp"


template<typename NumericType>
NumericType compute_radii(size_t i_idx, std::vector<NumericType> xs) {
    NumericType sum = 0;
    for(auto j=0; j < xs.size(); j++) {
        NumericType dx = xs[i_idx] - 0.5*xs[j];
        sum += dx*dx;
    }
    return sum;
}

template<typename NumericType>
NumericType energy(
    NumericType ri,
    NumericType rj) {
    NumericType dx = ri - 1.5*rj;
    NumericType d2ij = dx*dx;
    return d2ij;
}

template<typename NumericType>
std::vector<NumericType> dRi_dxall(size_t i_idx, std::vector<NumericType> xs) {
    std::vector<NumericType> grads(xs.size());
    for(auto j=0; j < xs.size(); j++) {
        NumericType dx = xs[i_idx] - 0.5*xs[j];
        grads[i_idx] += 2*dx*1;
        grads[j] += 2*dx*(-0.5);
    }
    return grads;
}


template<typename NumericType>
NumericType outer_loop(std::vector<NumericType> xs) {
    NumericType nrg_sum = 0;
    
    double step = 1e-100;

    std::vector<NumericType> R(xs.size(), 0);
    for(size_t i=0; i < xs.size(); i++) {
        R[i] = compute_radii(i, xs);
    }
    // std::cout << "---" << std::endl;
    std::vector<NumericType> dE_dxs(xs.size(), 0.0);

    // derivative of energy with respect to each radii

    // for(size_t i=0; i < xs.size(); i++) {
    //     std::complex<NumericType> cR_i = std::complex<NumericType>(R[i], step);
    //     std::complex<NumericType> cR_j = std::complex<NumericType>(R[j], 0);
    //     auto nrg = energy(cR_i, cR_j);
    //     dE_dRi[i] = nrg.imag()/step;

    //     cR_i = std::complex<NumericType>(R[i], 0);
    //     cR_j = std::complex<NumericType>(R[j], step);
    //     nrg = energy(cR_i, cR_j);
    //     dE_dRj[i] = nrg.imag()/step;
    // }
    
    std::complex<NumericType> cR_i;
    std::complex<NumericType> cR_j;
    std::vector<std::complex<NumericType> > cxs;

    std::vector<NumericType> dRs(xs.size(), 0.0);


    std::set<NumericType> Ri_uniques;
    std::set<NumericType> Rj_uniques;

    std::vector<std::vector<NumericType> > all_dris; // [N, N]
    for(size_t i=0; i < xs.size(); i++) {
        all_dris.push_back(dRi_dxall(i, xs));
    }

    std::vector<NumericType> dE_dRi_sum(xs.size(), 0);
    std::vector<NumericType> dE_dRj_sum(xs.size(), 0);

    // O(N^2) method. The tricky part is that we'd need to compute the Hessian in an efficient way as well.

    // compute the derivative of the energy with respect to each radii
    for(size_t i=0; i < xs.size(); i++) {
        for(size_t j=0; j < xs.size(); j++) {

            // each interaction contributes not only to xi and xj, but also xk indirectly:
            // dE/dRi (dRi/dxi + dRi/dxj + dRi/dxk + ... + dRi/dxn)
            // dE/dRj (dRj/dxi + dRj/dxj + dRj/dxk + ... + dRj/dxn)

            auto ixn_nrg = energy(R[i], R[j]);
            nrg_sum += ixn_nrg;

            // this part is cheap to compute but they're unique
            cR_i = std::complex<NumericType>(R[i], step);
            cR_j = std::complex<NumericType>(R[j], 0);
            auto dE_dRi = energy(cR_i, cR_j).imag()/step;


            cR_i = std::complex<NumericType>(R[i], 0);
            cR_j = std::complex<NumericType>(R[j], step);
            auto dE_dRj = energy(cR_i, cR_j).imag()/step;

            dE_dRi_sum[i] += dE_dRi;
            dE_dRj_sum[j] += dE_dRj;

        }
    }

    // accumulate the energies 
    for(size_t i=0; i < xs.size(); i++) {
        for(size_t k=0; k < xs.size(); k++) {

            auto drik = all_dris[i][k];
            dE_dxs[k] += drik*dE_dRi_sum[i] + drik*dE_dRj_sum[i];


            // even the analytic expression is O(N^3)
            // auto dri = all_dris[i];
            // auto drj = all_dris[j];


            // for(size_t k=0; k < xs.size(); k++) {
                // dE_dxs[k] += dE_dRi * dri[k] + dE_dRj * drj[k];
            // }

            // dE_dxs[k] += dE_dRi*drik_sum + dE_dRj*drjk_sum;
            
            // compute dRi_dxk - O(N^4) even less efficient!
            // for(size_t k=0; k < xs.size(); k++) {
            //     cxs = timemachine::convert_to_complex<NumericType>(xs);
            //     cxs[k] = std::complex<NumericType>(xs[k], step);
            //     auto dRi_dk = compute_radii(i, cxs).imag()/step;
            //     auto dRj_dk = compute_radii(j, cxs).imag()/step;

            //     Ri_uniques.insert(dRi_dk);
            //     Rj_uniques.insert(dRj_dk);
            //     std::cout << i << " " << j << " | " << k << " " << dRi_dk << " " << dRj_dk << std::endl;
            //     dE_dxs[k] += dE_dRi*dRi_dk + dE_dRj*dRj_dk;
            // }

        }
    }

    for(auto i : Ri_uniques) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    for(auto j : Rj_uniques) {
        std::cout << j << " ";
    }
    std::cout << std::endl;

    for(size_t i=0; i < dE_dxs.size(); i++){
        std::cout << dE_dxs[i] << " " << std::endl;
    }

    std::cout << "NRG" << nrg_sum << std::endl;

    return nrg_sum;
}


int main() {

    // std::vector<double> xs({1.0, 2.3, 0.4, -0.3, 1.2}); // only O(N^2) unique ones
    std::vector<double> xs({1.0, 2.3, 0.4});
    // std::vector<double> xs({1.0, 2.3});
    outer_loop(xs);


    // double step = 1e-100;
    // std::vector<std::complex<double> > cxs = timemachine::convert_to_complex<double>(xs);
    // for(auto i=0; i < xs.size(); i++) {
    //     cxs[i] = std::complex<double>(xs[i], step);
    //     std::cout << outer_loop(cxs).imag()/step << std::endl;
    //     cxs[i] = std::complex<double>(xs[i], 0);
    // }

    
}