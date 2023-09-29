#pragma once

#include "types.hpp"

void accumulate_energy(
    int N,
    const EnergyType *__restrict__ d_input_buffer, // [N]
    EnergyType *__restrict d_u_buffer,             // [1]
    cudaStream_t stream);
