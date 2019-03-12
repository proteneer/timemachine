#include "context.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

template <typename NumericType>
void Context<NumericType>::step(bool inference) {

    int N = integrator_->num_atoms();
    int P = integrator_->num_params();

    NumericType *de = integrator_->get_device_energy();
	NumericType *dg = integrator_->get_device_grads();


    gpuErrchk(cudaMemset(de, 0, sizeof(NumericType)));
    gpuErrchk(cudaMemset(dg, 0, N*3*sizeof(NumericType)));

    NumericType *dh = nullptr;
    NumericType *dm = nullptr;
    if(!inference) {
        dh = integrator_->get_device_hessians();
        dm = integrator_->get_device_mixed_partials();
        gpuErrchk(cudaMemset(dh, 0, N*3*N*3*sizeof(NumericType)));
        gpuErrchk(cudaMemset(dm, 0, P*N*3*sizeof(NumericType)));     
    }

    NumericType *d_coords = integrator_->get_device_coords();

    for(size_t i=0; i < energies_.size(); i++) {
    	EnergyGPU<NumericType>* nrg = energies_[i];

    	nrg->total_derivative(
    		N,
    		P,
    		d_coords,
    		de,
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