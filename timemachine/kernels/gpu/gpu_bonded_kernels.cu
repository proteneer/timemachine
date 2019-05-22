#include <cstdio>
#include "bonded_kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename RealType>
void harmonic_bond_hmp_gpu(
    const int num_atoms,
    const int num_params,
    const RealType *coords,
    const RealType *params,
    const RealType *dxdps,
    const int num_bonds,
    const int *bond_idxs,
    const int *param_idxs,
    RealType *grads,
    RealType *hmps) {

    RealType* d_coords;
    RealType* d_params;
    RealType* d_dxdps;
    RealType* d_grads;
    RealType* d_hmps;

    int* d_bond_idxs;
    int* d_param_idxs;

    const auto N = num_atoms;
    const auto P = num_params;
    const auto B = num_bonds;
    
    gpuErrchk(cudaMalloc((void**)&d_coords, N*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_params, P*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_dxdps, P*N*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_grads, N*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_hmps, P*N*3*sizeof(RealType)));

    gpuErrchk(cudaMemcpy(d_coords, coords, N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, params, P*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dxdps, dxdps, P*N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_grads, grads, N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_hmps, hmps, P*N*3*sizeof(RealType), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_bond_idxs, B*2*sizeof(*d_bond_idxs)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs, B*2*sizeof(*d_param_idxs)));
    gpuErrchk(cudaMemcpy(d_bond_idxs, bond_idxs, B*2*sizeof(*d_bond_idxs), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs, param_idxs, B*2*sizeof(*d_param_idxs), cudaMemcpyHostToDevice));

    size_t tpb = 32;
    size_t n_blocks = (num_bonds + tpb - 1) / tpb;

    dim3 dimBlock(tpb);
    dim3 dimGrid(n_blocks, num_params); // x, y

    harmonic_bond_hmp<<<dimGrid, dimBlock>>>(
      num_atoms,
      num_params,
      d_coords,
      d_params,
      d_dxdps,
      num_bonds,
      d_bond_idxs,
      d_param_idxs,
      d_grads,
      d_hmps);

    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(grads, d_grads, N*3*sizeof(RealType), cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(hmps, d_hmps, P*N*3*sizeof(RealType), cudaMemcpyDeviceToHost))

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_params));
    gpuErrchk(cudaFree(d_dxdps));
    gpuErrchk(cudaFree(d_grads));
    gpuErrchk(cudaFree(d_hmps));

    gpuErrchk(cudaFree(d_bond_idxs));
    gpuErrchk(cudaFree(d_param_idxs));

};

// instantiate explicitly
template void harmonic_bond_hmp_gpu<float>(
    const int num_atoms,
    const int num_params,
    const float *coords,
    const float *params,
    const float *dxdps,
    const int num_bonds,
    const int *bond_idxs,
    const int *param_idxs,
    float *grads,
    float *hmps);

template void harmonic_bond_hmp_gpu<double>(
    const int num_atoms,
    const int num_params,
    const double *coords,
    const double *params,
    const double *dxdps,
    const int num_bonds,
    const int *bond_idxs,
    const int *param_idxs,
    double *grads,
    double *hmps);