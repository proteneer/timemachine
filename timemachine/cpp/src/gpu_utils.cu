#include "gpu_utils.cuh"

namespace timemachine {

int round_up_even(int count) { return count + (count % 2); }

curandStatus_t templateCurandNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev) {
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

curandStatus_t
templateCurandNormal(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev) {
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

curandStatus_t templateCurandUniform(curandGenerator_t generator, float *outputPtr, size_t n) {
    return curandGenerateUniform(generator, outputPtr, n);
}

curandStatus_t templateCurandUniform(curandGenerator_t generator, double *outputPtr, size_t n) {
    return curandGenerateUniformDouble(generator, outputPtr, n);
}

void __global__ k_initialize_curand_states(const int count, const int seed, curandState_t *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < count) {
        curand_init(seed + idx, 0, 0, &states[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
