#pragma once

namespace timemachine {

void accumulate_energy(
    int N,
    const __int128 *__restrict__ d_input_buffer, // [N]
    __int128 *__restrict d_u_buffer,             // [1]
    cudaStream_t stream);

} // namespace timemachine
