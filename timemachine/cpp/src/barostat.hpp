// this file implements the MonteCarlo barostat from that of OpenMM
#pragma once

#include "bound_potential.hpp"
#include "curand.h"
#include <random>
#include <vector>

namespace timemachine {

//class Barostat {
//
//public:
//
//    virtual void inplace_move(
//        double *d_x,
//        double *d_box,
//        const double lambda
//    );
//
//};

class MonteCarloBarostat {

public:
    MonteCarloBarostat(
        const int N,
        const double pressure,    // in bar
        const double temperature, // in kelvin
        std::vector<std::vector<int>> group_idxs,
        const int interval,
        std::vector<BoundPotential *> bps,
        const int seed);

    ~MonteCarloBarostat();

    // inplace_move() may modify d_x and d_box
    void inplace_move(double *d_x, double *d_box, const double lambda, cudaStream_t stream);

    void set_interval(const int interval);

    int get_interval();

    void set_pressure(const double pressure);

private:
    const int N_;

    void reset_counters();

    const std::vector<BoundPotential *> bps_;

    double pressure_;
    const double temperature_;
    int interval_;
    const int seed_;
    const std::vector<std::vector<int>> group_idxs_;

    // stuff that deals with RNG
    double *d_rand_;
    curandGenerator_t cr_rng_;

    // internals
    int step_;
    int num_grouped_atoms_;

    int *d_num_attempted_;
    int *d_num_accepted_;

    unsigned long long *d_u_buffer_;
    unsigned long long *d_u_after_buffer_;

    unsigned long long *d_init_u_;
    unsigned long long *d_final_u_;

    double *d_volume_;
    double *d_volume_delta_;
    double *d_length_scale_;
    double *d_volume_scale_;

    double *d_x_after_;
    double *d_box_after_;

    int *d_atom_idxs_;   // grouped index to atom coords
    int *d_mol_idxs_;    // grouped index to molecule index
    int *d_mol_offsets_; // Offset of molecules to determine size of mols

    double *d_sum_storage_;
    size_t d_sum_storage_bytes_;

    unsigned long long *d_centroids_; // Accumulate centroids in fix point to ensure deterministic behavior
};

} // namespace timemachine
