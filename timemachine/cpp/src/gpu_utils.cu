#include "gpu_utils.cuh"

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    float *outputPtr, size_t n, 
    float mean, float stddev) {
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

// #include <iostream> 
curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    double *outputPtr, size_t n, 
    double mean, double stddev) {
    // std::cout << "N DOUBLE" << n << std::endl;
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
