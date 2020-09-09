#pragma once

#include <vector>
// #include "potential.hpp"
#include "integrator.hpp"
#include "bound_potential.hpp"
#include "observable.hpp"

namespace timemachine {

class Context {

public:

    Context(
        int N,
        const double *x_0,
        const double *v_0,
        const double *box_0,
        // double lambda,
        Integrator *intg,
        std::vector<BoundPotential *> bps
    );

    ~Context();

    void add_observable(Observable *obs); // tbd: shared_ptr

    void step(double lambda);

    int num_atoms() const;

    double get_u_t() const;

    void get_du_dx_t(unsigned long long *out_buffer) const;

    void get_x_t(double *out_buffer) const;

    void get_v_t(double *out_buffer) const;

private:

    void compute(unsigned int flags);

    int step_;
    int N_; // number of particles

    double *d_x_t_; // coordinates
    double *d_v_t_; // velocities
    double *d_box_t_; // box vectors
    double *d_u_t_; // u (energy)
    // double lambda_; // (ytz): not a pointer!

    unsigned long long *d_du_dx_t_; // du/dx 

    Integrator *intg_;
    std::vector<Observable *> observables_;
    std::vector<BoundPotential *> bps_;


};

}