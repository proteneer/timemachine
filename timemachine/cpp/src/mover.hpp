#pragma once

#include <array>
#include <cuda_runtime.h>
#include <vector>

namespace timemachine {

// Base class for generic moves that can be accepted by Context
class Mover {

public:
    virtual ~Mover(){};

    virtual void set_interval(const int interval) = 0;

    virtual int get_interval() = 0;

    virtual std::array<std::vector<double>, 2> move_host(const int N, const double *h_x, const double *h_box);

    virtual void move(const int N, double *d_x, double *d_box, cudaStream_t stream) = 0;
};

} // namespace timemachine
