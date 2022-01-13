#include "barostat.hpp"
#include "bound_potential.hpp"
#include "integrator.hpp"
#include <array>
#include <vector>

namespace timemachine {

class Context {

public:
    Context(
        int N,
        const double *x_0,
        const double *v_0,
        const double *box_0,
        Integrator *intg,
        std::vector<BoundPotential *> bps,
        MonteCarloBarostat *barostat = nullptr);

    ~Context();

    void step(double lambda);

    std::array<std::vector<double>, 3>
    multiple_steps(const std::vector<double> &lambda_schedule, int store_du_dl_interval, int store_x_interval);

    std::array<std::vector<double>, 3> multiple_steps_U(
        const double lambda,
        const int n_steps,
        const std::vector<double> &lambda_windows, // which lambda windows we want to evaluate U at
        int store_u_interval,
        int store_x_interval);

    int num_atoms() const;

    void get_du_dx_t_minus_1(unsigned long long *out_buffer) const;

    void set_x_t(const double *in_buffer);

    void get_x_t(double *out_buffer) const;

    void get_v_t(double *out_buffer) const;

    void get_box(double *out_buffer) const;

private:
    MonteCarloBarostat *barostat_;

    void _step(double lambda, unsigned long long *du_dl_out);

    void _step_equilibrium(double lambda, unsigned long long *du_dl_out);

    int step_;
    int N_; // number of particles

    double *d_x_t_;   // coordinates
    double *d_v_t_;   // velocities
    double *d_box_t_; // box vectors

    unsigned long long *d_du_dx_t_;      // du/dx [N,3]
    unsigned long long *d_du_dl_buffer_; // du/dl [N]
    unsigned long long *d_u_buffer_;     // u [N]
    double *d_sum_storage_;
    size_t d_sum_storage_bytes_;

    Integrator *intg_;
    std::vector<BoundPotential *> bps_;
};

} // namespace timemachine
