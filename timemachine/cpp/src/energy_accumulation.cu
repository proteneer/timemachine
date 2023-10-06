#include "energy_accumulation.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

void accumulate_energy(
    int N,
    const __int128 *__restrict__ d_input_buffer, // [N]
    __int128 *__restrict d_u_buffer,             // [1]
    cudaStream_t stream) {
    const static unsigned int THREADS_PER_BLOCK = 512;
    k_accumulate_energy<THREADS_PER_BLOCK><<<1, THREADS_PER_BLOCK, 0, stream>>>(N, d_input_buffer, d_u_buffer);
    gpuErrchk(cudaPeekAtLastError());
}

} // namespace timemachine
