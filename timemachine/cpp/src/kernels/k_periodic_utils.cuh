#pragma once

__device__ double apply_delta(double delta, double box) {
    return delta - floor(delta/box+0.5)*box;
}
