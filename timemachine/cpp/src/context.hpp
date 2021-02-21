#pragma once

#include <vector>
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
        Integrator *intg,
        std::vector<BoundPotential *> bps
    );

    ~Context();

    void add_observable(Observable *obs); // tbd: shared_ptr

    void step(double lambda);

    std::vector<double> multiple_steps(
        const std::vector<double> &lambda_schedule,
        int store_du_dl_freq=0
    );

    int num_atoms() const;

    void get_du_dx_t_minus_1(unsigned long long *out_buffer) const;

    void get_x_t(double *out_buffer) const;

    void get_v_t(double *out_buffer) const;

private:

    void _step(double lambda, double *du_dl_buffer);

    int step_;
    int N_; // number of particles

    double *d_x_t_; // coordinates
    double *d_v_t_; // velocities
    double *d_box_t_; // box vectors

    unsigned long long *d_du_dx_t_; // du/dx

    Integrator *intg_;
    std::vector<Observable *> observables_;
    std::vector<BoundPotential *> bps_;

};

}