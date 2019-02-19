#include "context.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

template <typename NumericType>
void Context<NumericType>::step() {

	NumericType *dg = integrator_->get_device_grads();
	NumericType *dh = integrator_->get_device_hessians();
	NumericType *dm = integrator_->get_device_mixed_partials();

	int N = integrator_->num_atoms();
	int P = integrator_->num_params();

    gpuErrchk(cudaMemset(dg, 0, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(dh, 0, N*3*N*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(dm, 0, P*N*3*sizeof(NumericType)));

    NumericType *d_coords = integrator_->get_device_coords();

    for(size_t i=0; i < energies_.size(); i++) {
    	EnergyGPU<NumericType>* nrg = energies_[i];

    	nrg->total_derivative(
    		N,
    		P,
    		d_coords,
    		nullptr,
    		dg,
    		dh,
    		dm);
    }

    integrator_->step_gpu(
    	dg, dh, dm
    );

}

}

template class timemachine::Context<double>;
template class timemachine::Context<float>;