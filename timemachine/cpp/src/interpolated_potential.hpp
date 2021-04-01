#pragma once

#include "potential.hpp"

#include <memory>
#include <typeinfo>

namespace jankmachine {

class InterpolatedPotential : public Potential {

private:

    std::shared_ptr<Potential> u_;

    double *d_p_interpolated_;

    size_t d_sum_storage_bytes_;
    double *d_sum_storage_buffer_;
    double *d_sum_storage_out_;

    double *d_du_dp_buffer_;
    double *d_du_dl_buffer_;

public: 

    InterpolatedPotential(
        std::shared_ptr<Potential> u,
        int N,
        int P
    );

    ~InterpolatedPotential();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) override;

};

}
