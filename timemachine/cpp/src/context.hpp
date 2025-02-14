#include "barostat.hpp"
#include "bound_potential.hpp"
#include "integrator.hpp"
#include "local_md_potentials.hpp"
#include "mover.hpp"
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
        std::shared_ptr<Integrator> intg,
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        std::vector<std::shared_ptr<Mover>> &movers);

    ~Context();

    void step();
    void initialize();
    void finalize();

    void multiple_steps(const int n_steps, const int n_samples, double *h_x, double *h_box);

    void multiple_steps_local(
        const int n_steps,
        const std::vector<int> &local_idxs,
        const int n_samples,
        const double radius,
        const double k,
        const int seed,
        double *h_x,
        double *h_box);

    void multiple_steps_local_selection(
        const int n_steps,
        const int reference_idx,
        const std::vector<int> &selection_idxs,
        const int n_samples,
        const double radius,
        const double k,
        double *h_x,
        double *h_box);

    int num_atoms() const;

    void set_x_t(const double *in_buffer);

    void get_x_t(double *out_buffer) const;

    void set_v_t(const double *in_buffer);

    void get_v_t(double *out_buffer) const;

    void set_box(const double *in_buffer);

    void get_box(double *out_buffer) const;

    void setup_local_md(double temperature, bool freeze_reference);

    std::shared_ptr<Integrator> get_integrator() const;

    std::vector<std::shared_ptr<BoundPotential>> get_potentials() const;

    std::vector<std::shared_ptr<Mover>> get_movers() const;

    std::shared_ptr<MonteCarloBarostat<float>> get_barostat() const;

private:
    int N_; // number of particles

    std::vector<std::shared_ptr<Mover>> movers_;

    void _step(std::vector<std::shared_ptr<BoundPotential>> &bps, unsigned int *d_atom_idxs, const cudaStream_t stream);

    double _get_temperature();

    void _ensure_local_md_intialized();

    void _verify_coords_and_box(const double *coords_buffer, const double *box_buffer, cudaStream_t stream);

    int step_;

    cudaStream_t stream_;

    double *d_x_t_;   // coordinates
    double *d_v_t_;   // velocities
    double *d_box_t_; // box vectors

    std::shared_ptr<Integrator> intg_;
    std::vector<std::shared_ptr<BoundPotential>> bps_;
    std::vector<std::shared_ptr<BoundPotential>> nonbonded_pots_; // Potentials used to verify
    std::unique_ptr<LocalMDPotentials> local_md_pots_;
};

} // namespace timemachine
