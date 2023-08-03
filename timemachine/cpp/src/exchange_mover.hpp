#include <vector>
#include <array>

namespace timemachine {

class InsideOutsideExchangeMover {

public:
    InsideOutsideExchangeMover(
        double nb_beta,
        double nb_cutoff,
        const std::vector<double> &nb_params, // Nx4
        const std::vector<int> &water_idxs,   // Wx3
        const std::vector<int> &ligand_idxs,
        double beta,
        double radius);

    void get_water_groups(
        const std::vector<double> &coords,
        const std::array<double, 9> &box,
        const std::array<double, 3> &center,
        std::vector<int> &v1_mols,
        std::vector<int> &v2_mols) const;

    void swap_vi_into_vj_impl(
        int chosen_water,
        int N_i,
        int N_j,
        // const std::vector<int> &vi_mols,
        // const std::vector<int> &vj_mols,
        const std::vector<double> &coords,
        const std::array<double, 9> &box,
        const std::array<double, 3> &insertion_site,
        double vol_i,
        double vol_j,
        std::vector<double> &proposal_coords,
        double &log_prob) const;

    void swap_vi_into_vj(
        const std::vector<int> &vi_mols,
        const std::vector<int> &vj_mols,
        const std::vector<double> & coords,
        const std::array<double, 9> &box,
        const std::array<double, 3> &insertion_site,
        double vol_i,
        double vol_j,
        std::vector<double> &proposal_coords,
        double &log_prob) const;


    void propose(
        const std::vector<double> & coords,
        const std::array<double, 9> &box,
        std::vector<double> &proposal_coords,
        double &log_prob) const;

private:
    const double nb_beta_;
    const double nb_cutoff_;
    const std::vector<double> nb_params_; // Nx4
    const std::vector<int> water_idxs_;   // Wx3
    const std::vector<int> ligand_idxs_;
    const double beta_;
    const double radius_;

};

} // namespace timemachine
