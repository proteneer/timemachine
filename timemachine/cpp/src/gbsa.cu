#include <stdexcept>
#include <iostream> 
#include "fixed_point.hpp"
#include "gbsa.hpp"
#include "gbsa_jvp.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "k_gbsa.cuh"
#include "k_gbsa_jvp.cuh"

namespace timemachine {

template <typename RealType, int D>
GBSA<RealType, D>::GBSA(
    const std::vector<int> &charge_param_idxs, // [N]
    const std::vector<int> &atomic_radii_idxs, // [N]
    const std::vector<int> &scale_factor_idxs, // [E,2]
    double alpha,
    double beta,
    double gamma,
    double dielectric_offset,
    double surface_tension,
    double solute_dielectric,
    double solvent_dielectric,
    double probe_radius,
    double cutoff) :

    N_(charge_param_idxs.size()),
    alpha_(alpha),
    beta_(beta),
    gamma_(gamma),
    dielectric_offset_(dielectric_offset),
    surface_tension_(surface_tension),
    solute_dielectric_(solute_dielectric),
    solvent_dielectric_(solvent_dielectric),
    probe_radius_(probe_radius),
    cutoff_(cutoff) {

    gpuErrchk(cudaMalloc(&d_charge_param_idxs_, N_*sizeof(*d_charge_param_idxs_)));
    gpuErrchk(cudaMalloc(&d_scale_factor_idxs_, N_*sizeof(*d_scale_factor_idxs_)));
    gpuErrchk(cudaMalloc(&d_atomic_radii_idxs_, N_*sizeof(*d_atomic_radii_idxs_)));
    gpuErrchk(cudaMemcpy(d_charge_param_idxs_, &charge_param_idxs[0], N_*sizeof(*d_charge_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_factor_idxs_, &scale_factor_idxs[0], N_*sizeof(*d_scale_factor_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_atomic_radii_idxs_, &atomic_radii_idxs[0], N_*sizeof(*d_atomic_radii_idxs_), cudaMemcpyHostToDevice));

    // we probaly don't need *all* these buffers if we do just one pass, but they take up only
    // O(N) ram so we don't really care and just pre-allocate everything to keep things simple.
    // it also ensures that we can RAII properly.

    const int N = charge_param_idxs.size();

    gpuErrchk(cudaMalloc(&d_born_radii_buffer_, N*sizeof(*d_born_radii_buffer_)));
    gpuErrchk(cudaMalloc(&d_obc_buffer_, N*sizeof(*d_obc_buffer_)));
    gpuErrchk(cudaMalloc(&d_born_forces_buffer_, N*sizeof(*d_born_forces_buffer_)));

    gpuErrchk(cudaMalloc(&d_born_radii_buffer_jvp_, N*sizeof(*d_born_radii_buffer_jvp_)));
    gpuErrchk(cudaMalloc(&d_obc_buffer_jvp_, N*sizeof(*d_obc_buffer_jvp_)));
    gpuErrchk(cudaMalloc(&d_obc_ri_buffer_jvp_, N*sizeof(*d_obc_ri_buffer_jvp_)));
    gpuErrchk(cudaMalloc(&d_born_forces_buffer_jvp_, N*sizeof(*d_born_forces_buffer_jvp_)));


}

template <typename RealType, int D>
GBSA<RealType, D>::~GBSA() {

  gpuErrchk(cudaFree(d_charge_param_idxs_));
  gpuErrchk(cudaFree(d_atomic_radii_idxs_));
  gpuErrchk(cudaFree(d_scale_factor_idxs_));

  gpuErrchk(cudaFree(d_born_radii_buffer_)); // double or Surreal<double>
  gpuErrchk(cudaFree(d_obc_buffer_)); // double or Surreal<double>
  gpuErrchk(cudaFree(d_born_forces_buffer_)); // ull or Surreal<double>

  gpuErrchk(cudaFree(d_born_radii_buffer_jvp_)); // double or Surreal<double>
  gpuErrchk(cudaFree(d_obc_buffer_jvp_)); // double or Surreal<double>
  gpuErrchk(cudaFree(d_obc_ri_buffer_jvp_)); // double or Surreal<double>
  gpuErrchk(cudaFree(d_born_forces_buffer_jvp_)); // ull or Surreal<double>


};

// reference code
template<int D>
double compute_born_first_loop(
    const std::vector<double>& coords,
    const std::vector<double>& params,
    const std::vector<int>& charge_param_idxs,
    const std::vector<double>& born_radii,
    const double prefactor,
    const double cutoff,
    std::vector<double> &bornForces,
    std::vector<double> &out_forces,
    std::vector<double> &dU_dp) {

    const int numberOfAtoms = charge_param_idxs.size();
    const int N = numberOfAtoms;

    // const double soluteDielectric = solute_dielectric;
    // const double solventDielectric = solvent_dielectric;
    // double preFactor;

    // if (soluteDielectric != 0.0 && solventDielectric != 0.0) {
    //     preFactor = -screening*((1.0/soluteDielectric) - (1.0/solventDielectric));    
    // } else {
    //     preFactor = 0.0;
    // }
    // printf("preFactor %f\n", preFactor);
    std::vector<double> charge_derivs(N, 0);

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
 
        double partialChargeI = params[charge_param_idxs[atomI]];
        for (int atomJ = atomI; atomJ < numberOfAtoms; atomJ++) {

            double r2 = 0;
            double dxs[D] = {0};
            for(int d=0; d < D; d++) {
                double dx = coords[atomI*D+d] - coords[atomJ*D+d];
                dxs[d] = dx;
                r2 += dx*dx;
            }
            double r = sqrt(r2);
            double alpha2_ij          = born_radii[atomI]*born_radii[atomJ];
            double D_ij               = r2/(4.0*alpha2_ij);

            double expTerm            = exp(-D_ij);
            double denominator2       = r2 + alpha2_ij*expTerm; 
            double denominator        = sqrt(denominator2);

            double partialChargeJ     = params[charge_param_idxs[atomJ]];
            double Gpol               = (prefactor*partialChargeI*partialChargeJ)/denominator; 

            double dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  
            double dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;

            printf("%d %d dGpol_dalpha2_ij %f\n", atomI, atomJ, dGpol_dalpha2_ij);

            double energy = Gpol;

            double dE_dqi = prefactor*partialChargeJ/denominator;
            double dE_dqj = prefactor*partialChargeI/denominator;

            if (atomI != atomJ) {

                // TBD: determine what we should do with cutoff
                // energy -= partialChargeI*partialCharges[atomJ]/cutoff;
                bornForces[atomJ]        += dGpol_dalpha2_ij*born_radii[atomI];
                for(int d=0; d < D; d++) {
                    out_forces[atomI*D+d] += dxs[d]*dGpol_dr;
                    out_forces[atomJ*D+d] -= dxs[d]*dGpol_dr;
                }
            } else {
                dE_dqi *= 0.5;
                dE_dqj *= 0.5;
                energy *= 0.5;
            }

            charge_derivs[atomI]     += dE_dqi;
            charge_derivs[atomJ]     += dE_dqj;

            // obcEnergy         += energy;
            bornForces[atomI] += dGpol_dalpha2_ij*born_radii[atomJ];

        }
    }

    for(int i=0; i < charge_derivs.size(); i++) {
      // std::cout << "???" << charge_derivs[i] << std::endl;
        dU_dp[charge_param_idxs[i]] += charge_derivs[i];
    }

};

// ported over from OpenMM with minor corrections
// reference code
template<int D>
void compute_born_radii(
    const std::vector<double>& coords,
    const std::vector<double>& params,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<int>& scale_factor_idxs,
    const double dielectric_offset,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    const double cutoff,
    std::vector<double>& born_radii,
    std::vector<double>& obc_chain,
    std::vector<double>& obc_chain_ri) {

    int numberOfAtoms = atomic_radii_idxs.size();

    if(coords.size() / D != numberOfAtoms) {
        throw std::runtime_error("compute born radii number of atoms are inconsistent");
    }

    for (int i_idx = 0; i_idx < numberOfAtoms; i_idx++) {
      
       double radiusI         = params[atomic_radii_idxs[i_idx]];
       double offsetRadiusI   = radiusI - dielectric_offset;
       double radiusIInverse  = 1.0/offsetRadiusI;
       double sum             = 0.0;

       // HCT code
       for (int j_idx = 0; j_idx < numberOfAtoms; j_idx++) {

          if (j_idx != i_idx) {

             double r = 0;
             for(int d=0; d < D; d++) {
                double dx = coords[i_idx*D+d] - coords[j_idx*D+d];
                r += dx*dx;
             }
             r = sqrt(r);

             double offsetRadiusJ   = params[atomic_radii_idxs[j_idx]] - dielectric_offset; 
             double scaledRadiusJ   = offsetRadiusJ*params[scale_factor_idxs[j_idx]];
             double rScaledRadiusJ  = r + scaledRadiusJ;

             double rSubScaledRadiusJ =  r - scaledRadiusJ;
             if (offsetRadiusI < rScaledRadiusJ) {
                double rInverse = 1.0/r;

                double l_ij = 0;
                if(offsetRadiusI > abs(rSubScaledRadiusJ)) {
                  l_ij = offsetRadiusI;
                } else {
                  l_ij = abs(rSubScaledRadiusJ);
                }

                l_ij     = 1.0/l_ij;

                double u_ij     = 1.0/rScaledRadiusJ;

                double l_ij2    = l_ij*l_ij;
                double u_ij2    = u_ij*u_ij;
 
                double ratio    = log((u_ij/l_ij));
                double term     = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r)) {
                   term += 2.0*(radiusIInverse - l_ij);
                }
                sum += term;

               }
            }
         }
       sum              *= 0.5*offsetRadiusI;
       double sum2       = sum*sum;
       double sum3       = sum*sum2;
       double tanhSum    = tanh(alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3);
 
       born_radii[i_idx]      = 1.0/(1.0/offsetRadiusI - tanhSum/radiusI); 

       // dRi/dPsi
       obc_chain[i_idx]       = (alpha_obc - 2.0*beta_obc*sum + 3.0*gamma_obc*sum2); // !@#$ why did you move it here!
       obc_chain[i_idx]       = (1.0 - tanhSum*tanhSum)*obc_chain[i_idx]/radiusI; // this takes care of the radiusI prefactor
       obc_chain[i_idx]      *= born_radii[i_idx]*born_radii[i_idx];
       // dRi/dri
       obc_chain_ri[i_idx]    = 1.0/(offsetRadiusI*offsetRadiusI) - tanhSum/(radiusI*radiusI);
       obc_chain_ri[i_idx]   *= born_radii[i_idx]*born_radii[i_idx];

    }
}

// reference code
template <int D>
double reduce_born_forces(
    const std::vector<double>& coords,
    const std::vector<double>& params,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<double>& born_radii,
    const std::vector<double>& obc_chain,
    const std::vector<double>& obc_chain_ri,
    const double surface_tension, // surface area factor
    const double probe_radius,
    std::vector<double> &bornForces, // dU/Ri
    std::vector<double> &out_dU_dp
) {

    // constants
    const int numberOfAtoms = atomic_radii_idxs.size();
    const int N = numberOfAtoms;

    double obcEnergy = 0.0;

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {

        double radii_derivs = 0;
        if (born_radii[atomI] > 0.0) {
            double atomic_radii = params[atomic_radii_idxs[atomI]];
            double r            = atomic_radii + probe_radius;
            double ratio6       = pow(atomic_radii/born_radii[atomI], 6.0);
            double saTerm       = surface_tension*r*r*ratio6;
            obcEnergy          += saTerm;
            bornForces[atomI]  -= 6.0*saTerm/born_radii[atomI]; 
            double br2 = born_radii[atomI]*born_radii[atomI];
            double br4 = br2*br2;
            double br6 = br4*br2;
            radii_derivs += 2*pow(atomic_radii, 5)*surface_tension*(probe_radius + atomic_radii)*(3*probe_radius + 4*atomic_radii)/br6;
        }
        radii_derivs += bornForces[atomI] * obc_chain_ri[atomI];
        out_dU_dp[atomic_radii_idxs[atomI]] += radii_derivs;
        bornForces[atomI] *= obc_chain[atomI];
    }

}

// reference code
template <int D>
double compute_born_energy_and_forces(
    const std::vector<double>& coords,
    const std::vector<double>& params,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<int>& scale_factor_idxs,
    const std::vector<double>& born_radii,
    const std::vector<double>& obc_chain,
    const std::vector<double>& obc_chain_ri,
    const double dielectric_offset,
    // const double surface_tension, // surface area factor
    // const double probe_radius,
    const double cutoff,
    std::vector<double> &bornForces,
    std::vector<double> &out_forces,
    std::vector<double> &out_dU_dp) {

    // constants
    const int numberOfAtoms = atomic_radii_idxs.size();
    const int N = numberOfAtoms;

    const double dielectricOffset = dielectric_offset;
    const double cutoffDistance = cutoff;

    std::vector<double> dPsi_dx(N*D, 0);
    std::vector<double> dPsi_dri(N, 0);
    std::vector<double> dPsi_dsi(N, 0);
    // born forcesI will have been fully loaded by now
    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
 
       // radius w/ dielectric offset applied

       double radiusI        = params[atomic_radii_idxs[atomI]];
       double offsetRadiusI  = radiusI - dielectricOffset;
       double offsetRadiusI2 = offsetRadiusI*offsetRadiusI;
       double offsetRadiusI3 = offsetRadiusI2*offsetRadiusI;

       for (int atomJ = 0; atomJ < numberOfAtoms; atomJ++) {

          if (atomJ != atomI) {

             double r2 = 0;
             double dxs[D] = {0};
             for(int d=0; d < D; d++) {
                double dx = coords[atomI*D+d] - coords[atomJ*D+d];
                dxs[d] = dx;
                r2 += dx*dx;
             }
             double r = sqrt(r2);

             // radius w/ dielectric offset applied

             double radiusJ            = params[atomic_radii_idxs[atomJ]];
             double offsetRadiusJ      = radiusJ - dielectricOffset;
             double offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

             double scaleFactorJ       = params[scale_factor_idxs[atomJ]];
             double scaleFactorJ2      = scaleFactorJ*scaleFactorJ;
             double scaleFactorJ3      = scaleFactorJ2*scaleFactorJ;
             double scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
             double scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
             double rScaledRadiusJ     = r + scaledRadiusJ;
             double rScaledRadiusJ2    = rScaledRadiusJ*rScaledRadiusJ;
             double rScaledRadiusJ3    = rScaledRadiusJ2*rScaledRadiusJ;

             // dL/dr & dU/dr are zero (this can be shown analytically)
             // removed from calculation

             if (offsetRadiusI < rScaledRadiusJ) {

                // double l_ij          = offsetRadiusI > abs(rSubScaledRadiusJ) ? offsetRadiusI : abs(rSubScaledRadiusJ);
                //        l_ij          = 1.0/l_ij;
                // double u_ij          = 1.0/rScaledRadiusJ;
                // double l_ij2         = l_ij*l_ij;
                // double u_ij2         = u_ij*u_ij; 
                // double rInverse      = 1.0/r;
                // double r2Inverse     = rInverse*rInverse;
                // double t3            = 0.125*(1.0 + scaledRadiusJ2*r2Inverse)*(l_ij2 - u_ij2) + 0.25*log(u_ij/l_ij)*r2Inverse;

                // printf("%d t3 RHS: %.8f\n", atomI, t3);
                // double de            = bornForces[atomI]*t3*rInverse;

                // for(int d=0; d < D; d++) {
                //     dPsi_dx[atomI*D+d] -= dxs[d]*de;
                //     dPsi_dx[atomJ*D+d] += dxs[d]*de;
                // }

                // start manual derivative
                double de = 0; // derivative of Psi wrt the distance
                double dpsi_dri = 0;
                double dpsi_drj = 0;
                double dpsi_dsj = 0;

                double rSubScaledRadiusJ = r - scaledRadiusJ;
                double rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                double rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                // factor out as much as we can to outside of the conditional for reduce convergence
                if(offsetRadiusI > abs(rSubScaledRadiusJ)) {
                  double term = 0.5*(-offsetRadiusI)*(-0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + 1.0/rScaledRadiusJ + 1.0/(-offsetRadiusI) + 0.25*scaleFactorJ2*offsetRadiusJ2*(1/rScaledRadiusJ2 - 1/offsetRadiusI2)/r - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r);
                  de = -0.5*r/rScaledRadiusJ3 + (5.0/4.0)/rScaledRadiusJ2 - 0.25/offsetRadiusI2 + 0.5*scaleFactorJ2*offsetRadiusJ2/(r*rScaledRadiusJ3) - 0.5/(r*rScaledRadiusJ) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r2 - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r2;
                  dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + offsetRadiusI*(0.5*r/offsetRadiusI3 - 1/offsetRadiusI2 - 0.5*scaleFactorJ2*offsetRadiusJ2/(r*offsetRadiusI3) + 0.5/(r*offsetRadiusI)) - 1/rScaledRadiusJ + 1.0/offsetRadiusI + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*log(offsetRadiusI/rScaledRadiusJ)/r;
                  dpsi_drj = offsetRadiusI*(-0.5*r*scaleFactorJ/rScaledRadiusJ3 + scaleFactorJ/rScaledRadiusJ2 + 0.5*scaleFactorJ3*offsetRadiusJ2/(r*rScaledRadiusJ3) + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r - 0.5*scaleFactorJ/(r*rScaledRadiusJ));
                  dpsi_dsj = offsetRadiusI*(0.25*r*(2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3 + offsetRadiusJ/rScaledRadiusJ2 - 0.25*scaleFactorJ2*offsetRadiusJ2*(2*dielectricOffset - 2*radiusJ)/(r*rScaledRadiusJ3) + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*(-offsetRadiusJ)/(r*rScaledRadiusJ));
                  if(offsetRadiusI < (scaledRadiusJ - r)) {
                   de += 0;
                   dpsi_dri += 0;
                   dpsi_drj += 0;
                   dpsi_dsj += 0;
                  }

                } else {
                  double term = -0.5*(-offsetRadiusI)*(-0.25*r*(1/rSubScaledRadiusJ2 - 1/rScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rSubScaledRadiusJ2 + 1/rScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r);
                  de = 0.25*r*(-2/rScaledRadiusJ3 + 2/rSubScaledRadiusJ3) + (5.0/4.0)/rScaledRadiusJ2 - sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 - 0.25/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*(2/rScaledRadiusJ3 - 2/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(sign(rSubScaledRadiusJ)/rScaledRadiusJ - fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r2 - 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r2;
                  dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/rSubScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r;
                  dpsi_drj = offsetRadiusI*(0.25*r*(-2*scaleFactorJ/rScaledRadiusJ3 - 2*scaleFactorJ/rSubScaledRadiusJ3) + scaleFactorJ/rScaledRadiusJ2 + scaleFactorJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.25*scaleFactorJ2*offsetRadiusJ2*(2*scaleFactorJ/rScaledRadiusJ3 + 2*scaleFactorJ/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(-scaleFactorJ*sign(rSubScaledRadiusJ)/rScaledRadiusJ - scaleFactorJ*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                  dpsi_dsj = offsetRadiusI*(0.25*r*(-(-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 + (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3) + offsetRadiusJ/rScaledRadiusJ2 + offsetRadiusJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*((-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 - (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3)/r + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*rScaledRadiusJ*((-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rScaledRadiusJ + (-offsetRadiusJ)*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                  if (offsetRadiusI < (scaledRadiusJ - r)) {
                   de += 2.0*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                   dpsi_dri += -2.0/fabs(rSubScaledRadiusJ);
                   dpsi_drj += -2.0*scaleFactorJ*offsetRadiusI*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                   dpsi_dsj += 2.0*offsetRadiusI*(-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                  }
                }

                // is bornForces

                de *= 0.5*bornForces[atomI]*offsetRadiusI;
                dpsi_dri *= 0.5*bornForces[atomI];
                dpsi_drj *= 0.5*bornForces[atomI];
                dpsi_dsj *= 0.5*bornForces[atomI];

                dPsi_dri[atomI] += dpsi_dri;
                dPsi_dri[atomJ] += dpsi_drj;
                dPsi_dsi[atomJ] += dpsi_dsj;

                for(int d=0; d < D; d++) {
                    dPsi_dx[atomI*D+d] += (dxs[d]/r)*de;
                    dPsi_dx[atomJ*D+d] -= (dxs[d]/r)*de;
                }
             }
          }
       }

    }

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
      for(int d=0; d < D; d++) {
        out_forces[atomI*D+d] += dPsi_dx[atomI*D+d];
      }
    }


    for(int i=0; i < dPsi_dri.size(); i++) {
      out_dU_dp[atomic_radii_idxs[i]] += dPsi_dri[i];
    }

    for(int i=0; i < dPsi_dsi.size(); i++) {
      out_dU_dp[scale_factor_idxs[i]] += dPsi_dsi[i];
    }

}

template <typename RealType, int D>
void GBSA<RealType, D>::execute_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_coords_tangents,
    const double *d_params,
    unsigned long long *d_out_coords,
    double *d_out_coords_tangents,
    double *d_out_params_tangents
) {

    int tpb = 32;
    int B = (N_+tpb-1)/tpb;

    dim3 dimGrid(B, B, 1); // x, y, z dims

    double prefactor;
    if (solute_dielectric_ != 0.0 && solvent_dielectric_ != 0.0) {
        // prefactor = -screening_*((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
        prefactor = -((1.0/solute_dielectric_) - (1.0/solvent_dielectric_));
    } else {
        prefactor = 0.0;
    }

    // inference mode
    if(d_coords_tangents == nullptr) {
        gpuErrchk(cudaMemset(d_born_radii_buffer_, 0, N*sizeof(*d_born_radii_buffer_)));
        gpuErrchk(cudaMemset(d_obc_buffer_, 0, N*sizeof(*d_obc_buffer_)));
        gpuErrchk(cudaMemset(d_born_forces_buffer_, 0, N*sizeof(*d_born_forces_buffer_)));

        k_compute_born_radii_gpu<RealType, D><<<B, tpb>>>(
          N_,
          d_coords,
          d_params,
          d_atomic_radii_idxs_,
          d_scale_factor_idxs_,
          dielectric_offset_,
          alpha_,
          beta_,
          gamma_,
          cutoff_,
          d_born_radii_buffer_,
          d_obc_buffer_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_compute_born_first_loop_gpu<RealType, D><<<dimGrid, tpb>>>(
          N_,
          d_coords,
          d_params,
          d_charge_param_idxs_,
          d_born_radii_buffer_,
          prefactor,
          cutoff_,
          d_born_forces_buffer_, // output
          d_out_coords // ouput
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_reduce_born_forces<RealType, D><<<B, tpb>>>(
          N_,
          d_params,
          d_atomic_radii_idxs_,
          d_born_radii_buffer_,
          d_obc_buffer_,
          surface_tension_,
          probe_radius_,
          d_born_forces_buffer_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_compute_born_energy_and_forces<RealType, D><<<dimGrid, tpb>>>(
          N_,
          d_coords,
          d_params,
          d_atomic_radii_idxs_,
          d_scale_factor_idxs_,
          d_born_radii_buffer_,
          d_obc_buffer_,
          dielectric_offset_,
          cutoff_,
          d_born_forces_buffer_,
          d_out_coords
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

    } else {
        gpuErrchk(cudaMemset(d_born_radii_buffer_jvp_, 0, N*sizeof(*d_born_radii_buffer_jvp_)));
        gpuErrchk(cudaMemset(d_obc_buffer_jvp_, 0, N*sizeof(*d_obc_buffer_jvp_)));
        gpuErrchk(cudaMemset(d_obc_ri_buffer_jvp_, 0, N*sizeof(*d_obc_ri_buffer_jvp_)));
        gpuErrchk(cudaMemset(d_born_forces_buffer_jvp_, 0, N*sizeof(*d_born_forces_buffer_jvp_)));

        k_compute_born_radii_gpu_jvp<RealType, D><<<B, tpb>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_atomic_radii_idxs_,
            d_scale_factor_idxs_,
            dielectric_offset_,
            alpha_,
            beta_,
            gamma_,
            cutoff_,
            d_born_radii_buffer_jvp_,
            d_obc_buffer_jvp_,
            d_obc_ri_buffer_jvp_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_compute_born_first_loop_gpu_jvp<RealType, D><<<dimGrid, tpb>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_charge_param_idxs_,
            d_born_radii_buffer_jvp_,
            prefactor,
            cutoff_,
            d_born_forces_buffer_jvp_, // output
            d_out_coords_tangents, // ouput
            d_out_params_tangents // ouput
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_reduce_born_forces_jvp<RealType, D><<<B, tpb>>>(
            N_,
            d_params,
            d_atomic_radii_idxs_,
            d_born_radii_buffer_jvp_,
            d_obc_buffer_jvp_,
            d_obc_ri_buffer_jvp_,
            surface_tension_,
            probe_radius_,
            d_born_forces_buffer_jvp_,
            d_out_params_tangents
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        k_compute_born_energy_and_forces_jvp<RealType, D><<<dimGrid, tpb>>>(
            N_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_atomic_radii_idxs_,
            d_scale_factor_idxs_,
            d_born_radii_buffer_jvp_,
            d_obc_buffer_jvp_,
            d_obc_ri_buffer_jvp_,
            dielectric_offset_,
            cutoff_,
            d_born_forces_buffer_jvp_,
            d_out_coords_tangents,
            d_out_params_tangents
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

    }

}

template class GBSA<double, 4>;
template class GBSA<double, 3>;

template class GBSA<float, 4>;
template class GBSA<float, 3>;


}

