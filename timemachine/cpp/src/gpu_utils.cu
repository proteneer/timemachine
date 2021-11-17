#include "gpu_utils.cuh"

curandStatus_t templateCurandNormal(
    curandGenerator_t generator,
    float *outputPtr, size_t n,
    float mean, float stddev) {
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

curandStatus_t templateCurandNormal(
    curandGenerator_t generator,
    double *outputPtr, size_t n,
    double mean, double stddev) {
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
