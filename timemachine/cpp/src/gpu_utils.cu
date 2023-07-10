#include "gpu_utils.cuh"

int round_up_even(int count) { return count + (count % 2); }

curandStatus_t templateCurandNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev) {
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

curandStatus_t
templateCurandNormal(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev) {
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

void __global__ k_accumulate_energy(
    int N,
    const unsigned long long *__restrict__ input_buffer, // [N]
    unsigned long long *__restrict u_buffer              // [1]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    // Read out the value from the u_buffer
    unsigned long long accum = 0;
    while (idx < N) {
        accum += input_buffer[idx];
        idx += blockDim.x * gridDim.x;
    }
    // This could be a race condition if multiple `k_accumulate_energy` are running
    // on the same u_buffer
    u_buffer[0] += accum;
}
