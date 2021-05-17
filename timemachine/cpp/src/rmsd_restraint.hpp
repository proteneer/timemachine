#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class RMSDRestraint : public Potential {

private:

    const std::vector<int> h_atom_map_;
    const int N_;
    const double k_;
    unsigned long long *d_u_buf_;
    unsigned long long *d_du_dx_buf_;

public:

    RMSDRestraint(
        const std::vector<int> &atom_map,
        const int N_,
        const double k
    );

    ~RMSDRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx, // buffered
        double *d_du_dp,
        unsigned long long *d_du_dl,  // buffered
        unsigned long long *d_u,  // buffered
        cudaStream_t stream
    ) override;

};

}

