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
        double *coords_jvp,
        double *params_jvp) = 0;

};

class BasicStepper : public Stepper {

private:

    int count_;
    std::vector<Gradient<3> *>forces_;


public:

    BasicStepper(
        std::vector<Gradient<3> *> forces
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
        double *coords_jvp,
        double *params_jvp) override;

};

class LambdaStepper : public Stepper {

private:

    int count_;
    std::vector<Gradient <4> *>forces_;

    const std::vector<double> lambda_schedule_; // [T]

    double *d_coords_buffer_; // Nx4
    double *d_dx_tangent_buffer_; // Nx4
    double *d_coords_jvp_buffer_; // Nx4
    unsigned long long *d_forces_buffer_; // Nx4

    double *d_dw_dl_; // fixed and computed once
    int *d_lambda_flags_; // fixed [N]
    double *d_du_dl_; // [T]
    std::vector<double> du_dl_adjoint_; // [T], set later

public:

    void set_du_dl_adjoint(
        const int T,
        const double *adj);

    LambdaStepper(
        std::vector<Gradient <4> *> forces,
        const std::vector<double> &lambda_schedule,
        const std::vector<int> &lambda_flags
    );

    int get_T() const {
        return lambda_schedule_.size();
    }

    void get_du_dl(double *buf);

    virtual void forward_step(
        const int N,
        const int P,
        const double *coords, // Nx3
        const double *params, // Nx3
        unsigned long long *dx) override;

    void forward_step_host(
        const int N,
        const int P,
        const double *coords, // Nx3
        const double *params, // Nx3
        unsigned long long *dx);

    virtual void backward_step(
        const int N,
        const int P,
        const double *coords,
        const double *params,
        const double *dx_tangent,
        double *coords_jvp, // Nx3! need to truncate manually
        double *params_jvp) override;

    void backward_step_host(
        const int N,
        const int P,
        const double *coords,
        const double *params,
        const double *dx_tangent,
        double *coords_jvp, // Nx3! need to truncate manually
        double *params_jvp);

};


} // namespace