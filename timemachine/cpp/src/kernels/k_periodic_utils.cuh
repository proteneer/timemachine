#pragma once

template<typename T>
__device__ T apply_delta(T delta, double box) {
    return delta - floor(delta/box+0.5)*box;
}
