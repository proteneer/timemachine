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

    void step();
    void initialize();
    void finalize();

    std::array<std::vector<double>, 2> multiple_steps(const int n_steps, int store_x_interval);

    std::array<std::vector<double>, 3> multiple_steps_U(const int n_steps, int store_u_interval, int store_x_interval);

    std::array<std::vector<double>, 2> multiple_steps_local(
        const int n_steps,
        const std::vector<int> &local_idxs,
        const int burn_in,
        const int store_x_interval,
        const double radius,
        const double k,
        const int seed);

    int num_atoms() const;

    void set_x_t(const double *in_buffer);

    void get_x_t(double *out_buffer) const;

    void set_v_t(const double *in_buffer);

    void get_v_t(double *out_buffer) const;

    void get_box(double *out_buffer) const;

private:
    int N_; // number of particles

    MonteCarloBarostat *barostat_;

    void _step(std::vector<BoundPotential *> &bps, unsigned int *d_atom_idxs, const cudaStream_t stream);

    double _get_temperature();

    int step_;

    double *d_x_t_;   // coordinates
    double *d_v_t_;   // velocities
    double *d_box_t_; // box vectors

    unsigned long long *d_du_dx_t_;  // du/dx [N,3]
    unsigned long long *d_u_buffer_; // u [N]
    double *d_sum_storage_;
    size_t d_sum_storage_bytes_;

    Integrator *intg_;
    std::vector<BoundPotential *> bps_;
};

} // namespace timemachine
