// this file implements the MonteCarlo barostat from that of OpenMM
#pragma once

#include "bound_potential.hpp"
#include "curand.h"
#include "streamed_potential_runner.hpp"
#include <memory>
#include <vector>

namespace timemachine {

template <typename RealType> class MonteCarloBarostat {

public:
    MonteCarloBarostat(
        const int N,
        const double pressure,    // in bar
        const double temperature, // in kelvin
        std::vector<std::vector<int>> group_idxs,
        const int interval,
        std::vector<std::shared_ptr<BoundPotential>> bps,
        const int seed,
        const bool adaptive);

    ~MonteCarloBarostat();

    // inplace_move() may modify d_x and d_box
    void inplace_move(double *d_x, double *d_box, cudaStream_t stream);

    void set_interval(const int interval);

    int get_interval();

    double get_volume_scaling();

    void set_volume_scaling(const double scaling);

    void set_pressure(const double pressure);

    void set_adaptive_scaling(const bool adapative);

    bool get_adaptive_scaling();

private:
    const int N_;

    bool adaptive_; // Whether or no to adapt d_volume_scaling

    void reset_counters();

    std::vector<std::shared_ptr<BoundPotential>> bps_;

    RealType pressure_;
    const RealType temperature_;
    int interval_;
    const int seed_;
    const std::vector<std::vector<int>> group_idxs_;

    // stuff that deals with RNG
    RealType *d_rand_;
    curandGenerator_t cr_rng_;

    // internals
    int step_;
    int num_grouped_atoms_;

    int *d_num_attempted_;
    int *d_num_accepted_;

    __int128 *d_u_buffer_;
    __int128 *d_u_after_buffer_;

    __int128 *d_init_u_;
    __int128 *d_final_u_;

    RealType *d_volume_;
    RealType *d_volume_delta_;
    double *d_length_scale_;
    double *d_volume_scale_;

    double *d_x_after_;
    double *d_box_after_;

    int *d_atom_idxs_;   // grouped index to atom coords
    int *d_mol_idxs_;    // grouped index to molecule index
    int *d_mol_offsets_; // Offset of molecules to determine size of mols

    unsigned long long *d_centroids_; // Accumulate centroids in fixed point to ensure deterministic behavior

    StreamedPotentialRunner runner_;
};

template class MonteCarloBarostat<float>;
template class MonteCarloBarostat<double>;

} // namespace timemachine
