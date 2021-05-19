// this file implements the MonteCarlo barostat from that of OpenMM
#pragma once

#include <vector>
#include <random>
#include "bound_potential.hpp"

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

class MonteCarloBarostat{


public:

    MonteCarloBarostat(
        const int N,
        const double pressure, // in bar
        const double temperature, // in kelvin
        std::vector<std::vector<int> > group_idxs,
        const int interval,
        std::vector<BoundPotential *> bps,
        const int seed
    );


    ~MonteCarloBarostat();

    // inplace_move() may modify d_x and d_box
    void inplace_move(
        double *d_x,
        double *d_box,
        const double lambda
    );

    void set_interval(const int interval);

    void set_pressure(const double pressure);


private:

    void reset_counters();

    const std::vector<BoundPotential *> bps_;

    const int N_;
    double pressure_;
    const double temperature_;
    int interval_;
    const int seed_;
    const std::vector<std::vector<int> > group_idxs_;

    // stuff that deals with RNG
    std::mt19937 mt_; // Expected to be seeded with seed_
    std::uniform_real_distribution<double> dist_; // guaranteed unbiased

    // internals
    int step_;
    double volume_scale_;
    int num_attempted_;
    int num_accepted_;
    int num_grouped_atoms_;

    unsigned long long *d_u_before_;
    unsigned long long *d_u_after_;

    double *d_x_after_;

    double *d_box_after_;

    int *d_atom_idxs_; // grouped index to atom coords
    int *d_mol_idxs_; // grouped index to molecule index
    int *d_mol_offsets_; // Offset of molecules to determine size of mols
    double *d_centroids_;



};

}
