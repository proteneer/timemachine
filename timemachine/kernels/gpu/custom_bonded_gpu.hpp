#include <vector>

namespace timemachine {


template <typename RealType>
class HarmonicBond {

private:

    int* d_bond_idxs_;
    int* d_param_idxs_;
    int n_bonds_;

public:

    HarmonicBond(
        std::vector<int> bond_idxs,
        std::vector<int> param_idxs
    );

    ~HarmonicBond();

    // void set_params(const std::vector<RealType> &new_params);
    // std::vector<RealType> get_params() const;

    virtual void derivatives_host(
        const int num_atoms,
        const int num_params,
        const RealType *coords,
        const RealType *params,
        const RealType *dxdps,
        RealType *E,
        RealType *dE_dp,
        RealType *dE_dx,
        RealType *d2E_dxdp) const;

};


}