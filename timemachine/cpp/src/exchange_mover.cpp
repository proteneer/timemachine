#include "exchange_mover.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace timemachine {

double periodic_distance(const double *ri, const double *rj, double bx, double by, double bz) {
    double dx = ri[0] - rj[0];
    double dy = ri[1] - rj[1];
    double dz = ri[2] - rj[2];
    dx -= bx * (dx / bx + 0.5);
    dy -= by * (dy / by + 0.5);
    dz -= bz * (dz / bz + 0.5);
    return sqrt(dx * dx + dy * dy + dz * dz);
}

std::array<double, 3> compute_centroid(double *coords, int N) {

    double sum_x;
    double sum_y;
    double sum_z;

    for (int i = 0; i < N; i++) {
        sum_x += coords[i * 3 + 0];
        sum_y += coords[i * 3 + 1];
        sum_z += coords[i * 3 + 2];
    }

    return std::array<double, 3>({sum_x / N, sum_y / N, sum_z / N});
}

std::array<double, 3> v1_insertion(double radius, const std::array<double, 3> &center) {

    std::default_random_engine generator;
    std::normal_distribution<double> normal_dist;

    double x = normal_dist(generator);
    double y = normal_dist(generator);
    double z = normal_dist(generator);

    double n = sqrt(x * x + y * y + z * z);
    x /= n;
    y /= n;
    z /= n;

    std::uniform_real_distribution<double> uniform_dist;
    double c = cbrt(uniform_dist(generator));

    double new_x = x * c * radius + center[0];
    double new_y = y * c * radius + center[1];
    double new_z = z * c * radius + center[2];

    return std::array<double, 3>({new_x, new_y, new_z});
}

std::array<double, 3>
v2_insertion(double radius, const std::array<double, 3> &center, const std::array<double, 9> &box) {

    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform_dist;

    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    while (1) {
        double x = uniform_dist(generator) * bx;
        double y = uniform_dist(generator) * by;
        double z = uniform_dist(generator) * bz;
        std::array<double, 3> xyz({x, y, z});
        double dij = periodic_distance(&xyz[0], &center[0], bx, by, bz);
        if (dij >= radius) {
            return xyz;
        }
    }
}

double U_fn(
    const std::vector<double> &coords,
    const std::array<double, 9> &box,
    double nb_beta,
    double nb_cutoff,
    const std::vector<double> &nb_params,
    const std::array<int, 3> &a_idxs,
    const std::vector<int> &b_idxs) {

    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    double nrg = 0;

    for (size_t i = 0; i < a_idxs.size(); i++) {
        double xi = coords[i * 3 + 0];
        double yi = coords[i * 3 + 1];
        double zi = coords[i * 3 + 2];

        double qi = nb_params[i * 4 + 0];
        double sigi = nb_params[i * 4 + 1];
        double epsi = nb_params[i * 4 + 2];
        for (size_t j = 0; j < b_idxs.size(); j++) {
            double xj = coords[j * 3 + 0];
            double yj = coords[j * 3 + 1];
            double zj = coords[j * 3 + 2];

            double qj = nb_params[j * 4 + 0];
            double sigj = nb_params[j * 4 + 1];
            double epsj = nb_params[j * 4 + 2];

            double delta_x = xi - xj;
            double delta_y = yi - yj;
            double delta_z = zi - zj;

            delta_x -= bx * nearbyint(delta_x / bx);
            delta_y -= by * nearbyint(delta_y / by);
            delta_z -= bz * nearbyint(delta_z / bz);

            double dij = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

            if (dij < nb_cutoff) {

                double sig = sigi + sigj;
                double eps = epsi * epsj;
                double q = qi * qj;

                double sig6 = pow(sig / dij, 6);
                double sig12 = sig6 * sig6;

                double lj_nrg = 0;
                if (epsi != 0 && epsj != 0) {
                    lj_nrg = 4 * eps * (sig12 - sig6);
                }

                double es_nrg = q * erfc(nb_beta * dij) / dij;

                nrg += lj_nrg + es_nrg;
            }
        }
    }

    return nrg;
}

InsideOutsideExchangeMover::InsideOutsideExchangeMover(
    double nb_beta,
    double nb_cutoff,
    const std::vector<double> &nb_params, // Nx4
    const std::vector<int> &water_idxs,   // Wx3
    const std::vector<int> &ligand_idxs,
    double beta,
    double radius)
    : nb_beta_(nb_beta), nb_cutoff_(nb_cutoff), nb_params_(nb_params), // Nx4
      water_idxs_(water_idxs),                                         // Wx3
      ligand_idxs_(ligand_idxs), beta_(beta), radius_(radius) {}

// partition water into inside/outside regions
void InsideOutsideExchangeMover::get_water_groups(
    const std::vector<double> &coords,
    const std::array<double, 9> &box,
    const std::vector<double> &center,
    std::vector<int> &v1_mols,
    std::vector<int> &v2_mols) const {

    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    int N = coords.size() / 3;
    v1_mols.clear();
    v2_mols.clear();
    for (int i = 0; i < N; i++) {
        double dij = periodic_distance(&coords[i * 3], &center[0], bx, by, bz);
        if (dij < this->radius_) {
            v1_mols.push_back(i);
        } else {
            v2_mols.push_back(i);
        }
    }
}

void InsideOutsideExchangeMover::swap_vi_into_vj(
    const std::vector<int> &vi_mols,
    const std::vector<int> &vj_mols,
    const std::vector<double> &coords,
    const std::array<double, 9> &box,
    const std::array<double, 3> &center,
    bool insertion_mode,
    double vol_i,
    double vol_j) {

    int num_atoms = coords.size();

    int chosen_water = vi_mols[rand() % vi_mols.size()];

    // get indices of selected atoms
    std::array<int, 3> chosen_water_atom_idxs(
        {water_idxs_[chosen_water * 3 + 0], water_idxs_[chosen_water * 3 + 1], water_idxs_[chosen_water * 3 + 2]});

    // get old coordinates of the atoms
    std::array<double, 9> new_coords;
    for (int i = 0; i < 3; i++) {
        int atom_idx = chosen_water_atom_idxs[i];
        new_coords[i * 3 + 0] = coords[atom_idx * 3 + 0];
        new_coords[i * 3 + 1] = coords[atom_idx * 3 + 1];
        new_coords[i * 3 + 2] = coords[atom_idx * 3 + 2];
    }

    std::array<double, 3> new_center;
    if (insertion_mode) {
        new_center = v1_insertion(radius_, center);
    } else {
        new_center = v2_insertion(radius_, center, box);
    }

    // sample water
    // new_coords = ...
    // optimization - we can revert coordinates later to avoid having to copy this

    std::vector<double> trial_coords(coords);
    for (int i = 0; i < 3; i++) {
        int atom_idx = chosen_water_atom_idxs[i];
        trial_coords[atom_idx * 3 + 0] = new_coords[i * 3 + 0];
        trial_coords[atom_idx * 3 + 1] = new_coords[i * 3 + 1];
        trial_coords[atom_idx * 3 + 2] = new_coords[i * 3 + 2];
    }

    std::array<int, 3> a_idxs(chosen_water_atom_idxs);
    // set difference
    std::vector<int> b_idxs(num_atoms);
    for (int i = 0; i < num_atoms; i++) {
        b_idxs[i] = i;
    }
    b_idxs.erase(b_idxs.begin() + chosen_water * 3, b_idxs.begin() + (chosen_water + 1) * 3);

    double delta_U_insert = U_fn(trial_coords, box, nb_beta_, nb_cutoff_, nb_params_, a_idxs, b_idxs);
    double delta_U_delete = -U_fn(coords, box, nb_beta_, nb_cutoff_, nb_params_, a_idxs, b_idxs);
    double delta_U_total = delta_U_delete + delta_U_insert;

    size_t ni = vi_mols.size();
    size_t nj = vj_mols.size();

    double hastings_factor = log((ni * vol_j) / ((nj + 1) * vol_i));
    double log_p_accept = std::min(0.0, -beta_ * delta_U_total + hastings_factor);
}

} // namespace timemachine
