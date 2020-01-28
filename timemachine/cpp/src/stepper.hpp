#pragma once

#include <vector>

#include "gradient.hpp"

namespace timemachine {

template<typename RealType>
class Stepper {

public:

    virtual void forward_step(
        const int N,
        const int P,
        const RealType *coords,
        const RealType *params,
        unsigned long long *dx) = 0;

    virtual void backward_step(
        const int N,
        const int P,
        const RealType *coords,
        const RealType *params,
        const RealType *dx_tangent,
        RealType *coords_jvp,
        RealType *params_jvp) = 0;

};


template<typename RealType>
class BasicStepper : public Stepper<RealType> {

private:

    int count_;
    std::vector<Gradient <RealType, 3> *>forces_;


public:

    BasicStepper(
        std::vector<Gradient <RealType, 3> *> forces
    );

    virtual void forward_step(
        const int N,
        const int P,
        const RealType *coords,
        const RealType *params,
        unsigned long long *dx) override;

    virtual void backward_step(
        const int N,
        const int P,
        const RealType *coords,
        const RealType *params,
        const RealType *dx_tangent,
        RealType *coords_jvp,
        RealType *params_jvp) override;

};

template<typename RealType>
class LambdaStepper : public Stepper<RealType> {

private:

    int count_;
    std::vector<Gradient <RealType, 4> *>forces_;

    const std::vector<RealType> lambda_schedule_; // [T]

    int exponent_;

    RealType *d_coords_buffer_; // Nx4
    RealType *d_dx_tangent_buffer_; // Nx4
    RealType *d_coords_jvp_buffer_; // Nx4
    unsigned long long *d_forces_buffer_; // Nx4

    RealType *d_dw_dl_; // fixed and computed once
    int *d_lambda_flags_; // fixed [N]
    RealType *d_du_dl_; // [T]
    std::vector<RealType> du_dl_adjoint_; // [T], set later

public:

    void set_du_dl_adjoint(
        const int T,
        const RealType *adj);

    LambdaStepper(
        std::vector<Gradient <RealType, 4> *> forces,
        const std::vector<RealType> &lambda_schedule,
        const std::vector<int> &lambda_flags,
        const int exponent
    );

    int get_T() const {
        return lambda_schedule_.size();
    }

    void get_du_dl(RealType *buf);

    virtual void forward_step(
        const int N,
        const int P,
        const RealType *coords, // Nx3
        const RealType *params, // Nx3
        unsigned long long *dx) override;

    void forward_step_host(
        const int N,
        const int P,
        const RealType *coords, // Nx3
        const RealType *params, // Nx3
        unsigned long long *dx);

    virtual void backward_step(
        const int N,
        const int P,
        const RealType *coords,
        const RealType *params,
        const RealType *dx_tangent,
        RealType *coords_jvp, // Nx3! need to truncate manually
        RealType *params_jvp) override;

    void backward_step_host(
        const int N,
        const int P,
        const RealType *coords,
        const RealType *params,
        const RealType *dx_tangent,
        RealType *coords_jvp, // Nx3! need to truncate manually
        RealType *params_jvp);

};


} // namespace