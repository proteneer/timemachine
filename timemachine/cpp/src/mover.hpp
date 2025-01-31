#pragma once

#include <array>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace timemachine {

// Base class for generic moves that can be accepted by Context
class Mover {

protected:
    Mover(const int interval) : interval_(interval), step_(0) {};
    int interval_;
    int step_;

public:
    virtual ~Mover() {};

    // set_step is to deal with HREX where a mover may not be called during the singular frame being
    // generated.
    // TBD come up with an explicit, chainable API that works around this jank
    void set_step(const int step) {
        if (step < 0) {
            throw std::runtime_error("step must be at least 0");
        }
        this->step_ = step;
    }

    void set_interval(const int interval) {
        if (interval <= 0) {
            throw std::runtime_error("interval must be greater than 0");
        }
        this->interval_ = interval;
        // Clear the step, to ensure user can expect that in interval steps the barostat will trigger
        this->step_ = 0;
    }

    int get_interval() { return this->interval_; };

    virtual void move(const int N, double *d_x, double *d_box, cudaStream_t stream) = 0;

    virtual std::array<std::vector<double>, 2> move_host(const int N, const double *h_x, const double *h_box);
};

} // namespace timemachine
