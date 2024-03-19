#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "device_buffer.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace timemachine {

template <typename RealType> class FireMinimizer : public Integrator {

private:
    const int N_;
    const int n_min_;
    const int thread_blocks_;
    const RealType dt_max_;
    const RealType f_alpha_;
    const RealType f_inc_;
    const RealType f_dec_;

    DeviceBuffer<RealType> d_dt_;
    DeviceBuffer<RealType> d_alpha_;
    DeviceBuffer<int> d_step_;
    DeviceBuffer<int> d_accepted_;

    DeviceBuffer<unsigned long long>
        d_force_velo_dot_intermediate_; // Intermediate for whatever P is, dot product of forces and velocities
    DeviceBuffer<unsigned long long> d_force_velo_dot_; // The actual P value
    DeviceBuffer<unsigned long long> d_du_dx_old_;
    DeviceBuffer<unsigned long long> d_du_dx_;
    StreamedPotentialRunner runner_;

    DeviceBuffer<RealType> d_temp_buffer_;
    // Buffers for finding the max/summed value
    std::size_t temp_storage_bytes_;
    DeviceBuffer<char> d_temp_storage_buffer_;

public:
    FireMinimizer(
        const int N,
        const int n_min,
        const double dt_start,
        const double dt_max,
        const double f_inc,
        const double f_dec,
        const double alpha_start,
        const double f_alpha);

    virtual ~FireMinimizer();

    virtual void step_fwd(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t_,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void initialize(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void finalize(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;
};

} // end namespace timemachine
