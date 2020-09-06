#include "bound_potential.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

BoundPotential::BoundPotential(
    Potential *potential,
    std::vector<int> shape,
    const double *h_p
) : potential(potential),
    shape(shape) {

    int P = this->size();

    gpuErrchk(cudaMalloc(&d_p, P*sizeof(*d_p)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P*sizeof(*d_p), cudaMemcpyHostToDevice));

}

BoundPotential::~BoundPotential() {
    // only free the d_ps, but not the pure potentials themselves
    gpuErrchk(cudaFree(d_p));
}

int BoundPotential::size() const {
    if(shape.size() == 0) {
        return 0;
    }
    int total = 1;
    for(auto s : shape) {
        total *= s;
    }
    return total;
}

}
