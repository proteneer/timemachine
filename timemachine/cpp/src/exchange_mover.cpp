#include "exchange_mover.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>

namespace timemachine {

double periodic_distance(const double *ri, const double *rj, double bx, double by, double bz) {
    double dx = ri[0] - rj[0];
    double dy = ri[1] - rj[1];
    double dz = ri[2] - rj[2];
    dx -= bx * nearbyint(dx / bx);
    dy -= by * nearbyint(dy / by);
    dz -= bz * nearbyint(dz / bz);
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

    for (size_t a = 0; a < a_idxs.size(); a++) {

        size_t ii = a_idxs[a];

        double xi = coords[ii * 3 + 0];
        double yi = coords[ii * 3 + 1];
        double zi = coords[ii * 3 + 2];

        double qi = nb_params[ii * 4 + 0];
        double sigi = nb_params[ii * 4 + 1];
        double epsi = nb_params[ii * 4 + 2];

        for (size_t b = 0; b < b_idxs.size(); b++) {

            size_t jj = b_idxs[b];

            double xj = coords[jj * 3 + 0];
            double yj = coords[jj * 3 + 1];
            double zj = coords[jj * 3 + 2];

            double qj = nb_params[jj * 4 + 0];
            double sigj = nb_params[jj * 4 + 1];
            double epsj = nb_params[jj * 4 + 2];

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
    const std::array<double, 3> &center,
    std::vector<int> &v1_mols,
    std::vector<int> &v2_mols) const {

    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    v1_mols.clear();
    v2_mols.clear();

    for(size_t i=0; i < water_idxs_.size()/3; i++) {
    
        int o_idx = water_idxs_[i*3+0];
        int h1_idx = water_idxs_[i*3+1];
        int h2_idx = water_idxs_[i*3+2];

        std::array<double, 9> water_coords({
            coords[o_idx*3+0], coords[o_idx*3+1], coords[o_idx*3+2],
            coords[h1_idx*3+0], coords[h1_idx*3+1], coords[h1_idx*3+2],
            coords[h2_idx*3+0], coords[h2_idx*3+1], coords[h2_idx*3+2],
        });

        std::array<double, 3> centroid = compute_centroid(&water_coords[0], 3);

        double dij = periodic_distance(&centroid[0], &center[0], bx, by, bz);
        if (dij < this->radius_) {
            v1_mols.push_back(i);
        } else {
            v2_mols.push_back(i);
        }
    }
}

void InsideOutsideExchangeMover::propose(
    const std::vector<double> & coords,
    const std::array<double, 9> &box,
    std::vector<double> &proposal_coords,
    double &log_prob) const {


    std::vector<double> ligand_coords;

    for(size_t i=0; i < ligand_idxs_.size(); i++) {
        int idx = ligand_idxs_[i];
        ligand_coords.push_back(coords[idx*3+0]);
        ligand_coords.push_back(coords[idx*3+1]);
        ligand_coords.push_back(coords[idx*3+2]);
    }
    std::array<double, 3> ligand_centroid = compute_centroid(&ligand_coords[0], ligand_idxs_.size());
    
    std::vector<int> v1_mols, v2_mols;
    get_water_groups(coords, box, ligand_centroid, v1_mols, v2_mols);

    double PI = 3.14159265359;

    double vol_1 = (4 / 3) * PI * radius_ * radius_ * radius_;
    double vol_2 = box[0*3+0]*box[1*3+1]*box[2*3+2] - vol_1;

    std::array<double, 3> insertion_site_v1 = v1_insertion(radius_, ligand_centroid);
    std::array<double, 3> insertion_site_v2 = v2_insertion(radius_, ligand_centroid, box);

    int n1 = v1_mols.size();
    int n2 = v2_mols.size();

    if(n1 == 0 && n2 == 0) {
        throw std::runtime_error("jank");
    } else if(n1 > 0 && n2 == 0) {
        swap_vi_into_vj(
            v1_mols,
            v2_mols,
            coords,
            box,
            insertion_site_v2,
            vol_1,
            vol_2,
            proposal_coords,
            log_prob
        );
    } else if(n1 == 0 && n2 > 0) {
        swap_vi_into_vj(
            v2_mols,
            v1_mols,
            coords,
            box,
            insertion_site_v1,
            vol_2,
            vol_1,
            proposal_coords,
            log_prob
        );
    } else if(n1 > 0 && n2 > 0) {
        if(rand() % 2 == 0) {
            swap_vi_into_vj(
                v1_mols,
                v2_mols,
                coords,
                box,
                insertion_site_v2,
                vol_1,
                vol_2,
                proposal_coords,
                log_prob
            );
        } else {
            swap_vi_into_vj(
                v2_mols,
                v1_mols,
                coords,
                box,
                insertion_site_v1,
                vol_2,
                vol_1,
                proposal_coords,
                log_prob
            );
        }
    }



}

void InsideOutsideExchangeMover::swap_vi_into_vj(
    const std::vector<int> &vi_mols,
    const std::vector<int> &vj_mols,
    const std::vector<double> & coords,
    const std::array<double, 9> &box,
    const std::array<double, 3> &insertion_site,
    double vol_i,
    double vol_j,
    std::vector<double> &proposal_coords,
    double &log_prob) const {
    // chosen_water = np.random.choice(vi_mols)
    int chosen_water_idx = rand() % vi_mols.size();
    int chosen_water = vi_mols[chosen_water_idx];

    int N_i = vi_mols.size();
    int N_j = vj_mols.size();

    swap_vi_into_vj_impl(
        chosen_water,
        N_i,
        N_j,
        coords,
        box,
        insertion_site,
        vol_i,
        vol_j,
        proposal_coords,
        log_prob
    );
}

void InsideOutsideExchangeMover::swap_vi_into_vj_impl(
    int chosen_water,
    int N_i,
    int N_j,
    const std::vector<double> &coords,
    const std::array<double, 9> &box,
    const std::array<double, 3> &insertion_site,
    double vol_i,
    double vol_j,
    std::vector<double> &proposal_coords,
    double &log_prob) const {

    int num_atoms = coords.size() / 3;

    // get indices of selected atoms
    std::array<int, 3> chosen_water_atom_idxs({
        water_idxs_[chosen_water * 3 + 0],
        water_idxs_[chosen_water * 3 + 1],
        water_idxs_[chosen_water * 3 + 2]
    });


    // get old coordinates of the atoms
    std::array<double, 9> new_coords;
    for (int i = 0; i < 3; i++) {
        int atom_idx = chosen_water_atom_idxs[i];
        new_coords[i * 3 + 0] = coords[atom_idx * 3 + 0];
        new_coords[i * 3 + 1] = coords[atom_idx * 3 + 1];
        new_coords[i * 3 + 2] = coords[atom_idx * 3 + 2];
    }

    std::array<double, 3> new_coords_centroid = compute_centroid(&new_coords[0], 3);

    for (int i = 0; i < 3; i++) {
        new_coords[i * 3 + 0] = new_coords[i * 3 + 0] - new_coords_centroid[0] + insertion_site[0];
        new_coords[i * 3 + 1] = new_coords[i * 3 + 1] - new_coords_centroid[1] + insertion_site[1];
        new_coords[i * 3 + 2] = new_coords[i * 3 + 2] - new_coords_centroid[2] + insertion_site[2];
    }

    // sample water
    std::vector<double> trial_coords(coords);
    for (int i = 0; i < 3; i++) {
        int atom_idx = chosen_water_atom_idxs[i];
        trial_coords[atom_idx * 3 + 0] = new_coords[i * 3 + 0];
        trial_coords[atom_idx * 3 + 1] = new_coords[i * 3 + 1];
        trial_coords[atom_idx * 3 + 2] = new_coords[i * 3 + 2];
    }

    std::array<int, 3> a_idxs(chosen_water_atom_idxs);
    // set difference
    std::vector<int> b_idxs;
    for (int i = 0; i < num_atoms; i++) {
        bool found = false;
        for(auto a : a_idxs) {
            if(i==a) {
                found = true;
            }
        }
        if(!found) {
            b_idxs.push_back(i);
        }
    }

    double delta_U_insert = U_fn(trial_coords, box, nb_beta_, nb_cutoff_, nb_params_, a_idxs, b_idxs);
    double delta_U_delete = -U_fn(coords, box, nb_beta_, nb_cutoff_, nb_params_, a_idxs, b_idxs);
    double delta_U_total = delta_U_delete + delta_U_insert;

    double hastings_factor = log((N_i * vol_j) / ((N_j + 1) * vol_i));
    log_prob = std::min(0.0, -beta_ * delta_U_total + hastings_factor);
    proposal_coords = trial_coords;
}

} // namespace timemachine
