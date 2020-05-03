#pragma once

#include <vector>

#include "gradient.hpp"

namespace timemachine {

class Stepper {

private:

    std::vector<cudaStream_t> streams_;

protected:

    cudaStream_t get_stream(int force_idx);

    void sync_all_streams();

public:

    Stepper(int F);

    virtual ~Stepper();

    virtual void forward_step(
        const int N,
        const int P,
        const double *coords,
        const double *params,
        unsigned long long *dx) = 0;

    virtual void backward_step(
        const int N,
        const int P,
        const double *coords,
        const double *params,
        const double *dx_tangent,
        double *coords_jvp_primals,
        double *coords_jvp_tangents,
        double *params_jvp_primals,
        double *params_jvp_tangents) = 0;

};

class AlchemicalStepper : public Stepper {

private:

    int count_;
    std::vector<Gradient *>forces_;

    const std::vector<double> lambda_schedule_; // [T]
    double *d_du_dl_; // [T]
    double *d_energies_; // [T]
    std::vector<double> du_dl_adjoint_; // [T], set later


public:

    virtual ~AlchemicalStepper();

    AlchemicalStepper(
        std::vector<Gradient*> forces,
        const std::vector<double> &lambda_schedule
    );

    virtual void forward_step(
        const int N,
        const int P,
        const double *coords,
        const double *params,
        unsigned long long *dx) override;

    virtual void backward_step(
        const int N,
        const int P,
        const double *coords,
        const double *params,
        const double *dx_tangent,
        double *coords_jvp_primals,
        double *coords_jvp_tangents,
        double *params_jvp_primals,
        double *params_jvp_tangents) override;

    void get_du_dl(double *buf);
    void get_energies(double *buf);

    int get_T() const {
        return lambda_schedule_.size();
    }

    void set_du_dl_adjoint(
        const int T,
        const double *adj);

};


} // namespace