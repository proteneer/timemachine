#include <stdexcept>
#include <iostream> 
#include "fixed_point.hpp"
#include "gbsa.hpp"
#include "gbsa_jvp.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "k_gbsa.cuh"

namespace timemachine {

template <typename RealType, int D>
GBSAReference<RealType, D>::GBSAReference(
    const std::vector<int> &charge_param_idxs, // [N]
    const std::vector<int> &atomic_radii_idxs, // [N]
    const std::vector<int> &scale_factor_idxs, // [E,2]
    double alpha,
    double beta,
    double gamma,
    double dielectric_offset,
    double screening,
    double surface_tension,
    double solute_dielectric,
    double solvent_dielectric,
    double probe_radius,
    double cutoff) :
    charge_param_idxs_(charge_param_idxs),
    atomic_radii_idxs_(atomic_radii_idxs),
    scale_factor_idxs_(scale_factor_idxs),
    N_(charge_param_idxs.size()),
    alpha_(alpha),
    beta_(beta),
    gamma_(gamma),
    dielectric_offset_(dielectric_offset),
    screening_(screening),
    surface_tension_(surface_tension),
    solute_dielectric_(solute_dielectric),
    solvent_dielectric_(solvent_dielectric),
    probe_radius_(probe_radius),
    cutoff_(cutoff) {

    gpuErrchk(cudaMalloc(&d_scale_factor_idxs_, N_*sizeof(*d_scale_factor_idxs_)));
    gpuErrchk(cudaMalloc(&d_atomic_radii_idxs_, N_*sizeof(*d_atomic_radii_idxs_)));
    gpuErrchk(cudaMemcpy(d_scale_factor_idxs_, &scale_factor_idxs[0], N_*sizeof(*d_scale_factor_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_atomic_radii_idxs_, &atomic_radii_idxs[0], N_*sizeof(*d_atomic_radii_idxs_), cudaMemcpyHostToDevice));

}

// ported over from OpenMM with minor corrections
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
    // int N = numberOfAtoms;

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

template <int D>
double compute_born_energy_and_forces(
    const std::vector<double>& coords,
    const std::vector<double>& params,
    const std::vector<int>& charge_param_idxs,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<int>& scale_factor_idxs,
    const std::vector<double>& born_radii,
    const std::vector<double>& obc_chain,
    const std::vector<double>& obc_chain_ri,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    const double dielectric_offset,
    const double screening,
    const double surface_tension, // surface area factor
    const double solute_dielectric,
    const double solvent_dielectric,
    const double probe_radius,
    const double cutoff,
    std::vector<double> &out_forces,
    std::vector<double> &out_dU_dp
) {

    // constants
    const int numberOfAtoms = atomic_radii_idxs.size();
    const int N = numberOfAtoms;

    const double dielectricOffset = dielectric_offset;
    const double cutoffDistance = cutoff;
    const double soluteDielectric = solute_dielectric;
    const double solventDielectric = solvent_dielectric;
    double preFactor;

    if (soluteDielectric != 0.0 && solventDielectric != 0.0) {
        preFactor = -screening*((1.0/soluteDielectric) - (1.0/solventDielectric));    
    } else {
        preFactor = 0.0;
    }
    printf("preFactor %f\n", preFactor);

    double obcEnergy = 0.0;
    std::vector<double> bornForces(numberOfAtoms, 0.0);
    std::vector<double> atomic_radii_derivatives(N, 0);

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
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
            atomic_radii_derivatives[atomI] += 2*pow(atomic_radii, 5)*surface_tension*(probe_radius + atomic_radii)*(3*probe_radius + 4*atomic_radii)/br6;
        }
    }
 
    // ---------------------------------------------------------------------------------------

    // first main loop

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
          double Gpol               = (preFactor*partialChargeI*partialChargeJ)/denominator; 


          double dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  
          double dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;

          double energy = Gpol;

          double dE_dqi = preFactor*partialChargeJ/denominator;
          double dE_dqj = preFactor*partialChargeI/denominator;

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

          obcEnergy         += energy;
          bornForces[atomI] += dGpol_dalpha2_ij*born_radii[atomJ];

       }
    }

    // ---------------------------------------------------------------------------------------

    // second main loop
    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
      // order matters here
      atomic_radii_derivatives[atomI] += bornForces[atomI] * obc_chain_ri[atomI]; // do obc chain separately 
      bornForces[atomI] *= obc_chain[atomI]; // dU/dR*dR/dPsi
    }

    std::vector<double> dPsi_dx(N*D, 0);
    std::vector<double> dPsi_dri(N, 0);
    std::vector<double> dPsi_dsi(N, 0);

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

    // for(int i=0; i < dPsi_dri.size(); i++) {
    //   std::cout << "dPsi_dri: " << dPsi_dri[i]+atomic_radii_derivatives[i] << std::endl;
    // }

    for(int i=0; i < dPsi_dri.size(); i++) {
      // std::cout << "dPsi_dri parts: " << dPsi_dri[i] << " " << atomic_radii_derivatives[i] << std::endl;
      out_dU_dp[atomic_radii_idxs[i]] += dPsi_dri[i]+atomic_radii_derivatives[i];
    }

    for(int i=0; i < dPsi_dsi.size(); i++) {
      out_dU_dp[scale_factor_idxs[i]] += dPsi_dsi[i];
    }

    for(int i=0; i < charge_derivs.size(); i++) {
      // std::cout << "???" << charge_derivs[i] << std::endl;
      out_dU_dp[charge_param_idxs[i]] += charge_derivs[i];
    }

    std::cout << "energy" << obcEnergy << std::endl;
    // return obcEnergy;
}


template <typename RealType, int D>
void GBSAReference<RealType, D>::execute_first_order(
  const int N,
  const int P,
  const std::vector<double> &coords,
  const std::vector<double> &params,
  std::vector<double> &dU_dx,
  std::vector<double> &dU_dp
) {

    if(coords.size() != dU_dx.size()) {
      throw std::runtime_error("FATAL coords.size() != dU_dx.size()");
    }

    if(params.size() != dU_dp.size()) {
      throw std::runtime_error("FATAL params.size() != dU_dp.size()");
    }



    double* d_born_radii;
    double* d_obc_chain;
    double* d_obc_chain_ri;

    gpuErrchk(cudaMalloc(&d_born_radii, N*sizeof(*d_born_radii)));
    gpuErrchk(cudaMalloc(&d_obc_chain, N*sizeof(*d_obc_chain)));
    gpuErrchk(cudaMalloc(&d_obc_chain_ri, N*sizeof(*d_obc_chain_ri)));

    gpuErrchk(cudaMemset(d_born_radii, 0, N*sizeof(*d_born_radii)));
    gpuErrchk(cudaMemset(d_obc_chain, 0, N*sizeof(*d_obc_chain)));
    gpuErrchk(cudaMemset(d_obc_chain_ri, 0, N*sizeof(*d_obc_chain_ri)));

    // compute_born_radii<D>(
    //     coords,
    //     params,
    //     atomic_radii_idxs_,
    //     scale_factor_idxs_,
    //     dielectric_offset_,
    //     alpha_,
    //     beta_,
    //     gamma_,
    //     cutoff_,
    //     born_radii,
    //     obc_chain,
    //     obc_chain_ri
    // );

    double* d_coords;
    double* d_params;

    gpuErrchk(cudaMalloc(&d_coords, N*D*sizeof(*d_coords)));
    gpuErrchk(cudaMalloc(&d_params, N*D*sizeof(*d_params)));

    gpuErrchk(cudaMemcpy(d_coords, &coords[0], N*D*sizeof(*d_coords), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, &params[0], P*sizeof(*d_params), cudaMemcpyHostToDevice));

    int tpb = 32;
    int B = (N_+tpb-1)/tpb;

    compute_born_radii_gpu<double, D><<<B, tpb>>>(
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
        d_born_radii,
        d_obc_chain,
        d_obc_chain_ri
    );

    std::vector<double> h_born_radii(N, 0);
    std::vector<double> h_obc_chain(N, 0);
    std::vector<double> h_obc_chain_ri(N, 0);

    gpuErrchk(cudaMemcpy(&h_born_radii[0], d_born_radii, N*sizeof(*d_born_radii), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_obc_chain[0], d_obc_chain, N*sizeof(*d_obc_chain), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_obc_chain_ri[0], d_obc_chain_ri, N*sizeof(*d_obc_chain_ri), cudaMemcpyDeviceToHost));

    std::vector<double> out_forces(N*D, 0);

    // CPU
    compute_born_energy_and_forces<D>(
        coords,
        params,
        charge_param_idxs_,
        atomic_radii_idxs_,
        scale_factor_idxs_,
        h_born_radii,
        h_obc_chain,
        h_obc_chain_ri,
        alpha_,
        beta_,
        gamma_,
        dielectric_offset_,
        screening_,
        surface_tension_,
        solute_dielectric_,
        solvent_dielectric_,
        probe_radius_,
        cutoff_,
        dU_dx,
        dU_dp
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // for(int i=0; i< out_forces.size(); i++) {
    //     dU_dx[i] = static_cast<unsigned long long>((long long) (out_forces[i]*FIXED_EXPONENT));
    // }


}


template <typename RealType, int D>
void GBSAReference<RealType, D>::execute_second_order(
  const int N,
  const int P,
  const std::vector<double> &coords,
  const std::vector<double> &coords_tangent,
  const std::vector<double> &params,
  std::vector<double> &HvP,
  std::vector<double> &MvP
) {

    if(coords.size() != HvP.size()) {
      throw std::runtime_error("FATAL coords.size() != HvP.size()");
    }

    if(coords.size() != coords_tangent.size()) {
      throw std::runtime_error("FATAL coords.size() != coords_tangent.size()");
    }

    if(params.size() != MvP.size()) {
      throw std::runtime_error("FATAL params.size() != MvP.size()");
    }

    std::vector<Surreal<double> > born_radii(N, Surreal<double>(0, 0));
    std::vector<Surreal<double> > obc_chain(N, Surreal<double>(0, 0));
    std::vector<Surreal<double> > obc_chain_ri(N, Surreal<double>(0, 0));

    std::vector<Surreal<double> > dual_coords(N*D);

    for(int i=0; i < coords.size(); i++) {
      dual_coords[i].real = coords[i];
      dual_coords[i].imag = coords_tangent[i];
    }


    Surreal<double>* d_born_radii;
    Surreal<double>* d_obc_chain;
    Surreal<double>* d_obc_chain_ri;

    gpuErrchk(cudaMalloc(&d_born_radii, N*sizeof(*d_born_radii)));
    gpuErrchk(cudaMalloc(&d_obc_chain, N*sizeof(*d_obc_chain)));
    gpuErrchk(cudaMalloc(&d_obc_chain_ri, N*sizeof(*d_obc_chain_ri)));

    gpuErrchk(cudaMemset(d_born_radii, 0, N*sizeof(*d_born_radii)));
    gpuErrchk(cudaMemset(d_obc_chain, 0, N*sizeof(*d_obc_chain)));
    gpuErrchk(cudaMemset(d_obc_chain_ri, 0, N*sizeof(*d_obc_chain_ri)));


    Surreal<double>* d_coords;
    double* d_params;

    gpuErrchk(cudaMalloc(&d_coords, N*D*sizeof(*d_coords)));
    gpuErrchk(cudaMalloc(&d_params, N*D*sizeof(*d_params)));
    gpuErrchk(cudaMemcpy(d_coords, &dual_coords[0], N*D*sizeof(*d_coords), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, &params[0], P*sizeof(*d_params), cudaMemcpyHostToDevice));


    int tpb = 32;
    int B = (N_+tpb-1)/tpb;

    compute_born_radii_gpu_jvp<double, D><<<B, tpb>>>(
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
        d_born_radii,
        d_obc_chain,
        d_obc_chain_ri
    );

    std::vector<Surreal<double> > h_born_radii(N, Surreal<double>(0,0));
    std::vector<Surreal<double> > h_obc_chain(N, Surreal<double>(0,0));
    std::vector<Surreal<double> > h_obc_chain_ri(N, Surreal<double>(0,0));

    gpuErrchk(cudaMemcpy(&h_born_radii[0], d_born_radii, N*sizeof(*d_born_radii), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_obc_chain[0], d_obc_chain, N*sizeof(*d_obc_chain), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_obc_chain_ri[0], d_obc_chain_ri, N*sizeof(*d_obc_chain_ri), cudaMemcpyDeviceToHost));

    std::vector<double> out_forces(N*D, 0);

    // CPU
    compute_born_energy_and_forces_jvp<D>(
        dual_coords,
        params,
        charge_param_idxs_,
        atomic_radii_idxs_,
        scale_factor_idxs_,
        h_born_radii,
        h_obc_chain,
        h_obc_chain_ri,
        alpha_,
        beta_,
        gamma_,
        dielectric_offset_,
        screening_,
        surface_tension_,
        solute_dielectric_,
        solvent_dielectric_,
        probe_radius_,
        cutoff_,
        HvP,
        MvP
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // for(int i=0; i< out_forces.size(); i++) {
    //     dU_dx[i] = static_cast<unsigned long long>((long long) (out_forces[i]*FIXED_EXPONENT));
    // }


}

template <typename RealType, int D>
void GBSAReference<RealType, D>::execute_device(
    const int N,
    const int P,
    const double *d_coords,
    const double *d_coords_tangents,
    const double *d_params,
    unsigned long long *d_out_coords,
    double *d_out_coords_tangents,
    double *d_out_params_tangents
) {

    // compute born radii
    std::cout << N << " " << D << std::endl;
    std::cout << d_coords_tangents << std::endl;

    std::vector<double> coords(N*D, 0);
    std::vector<double> coords_tangents(N*D, 0);
    std::vector<double> params(P, 0);
    std::vector<unsigned long long> out_coords(N*D, 0);
    std::vector<double> out_coords_tangents(N*D, 0);
    std::vector<double> out_params_tangents(P, 0);

    gpuErrchk(cudaMemcpy(&coords[0], d_coords, sizeof(*d_coords)*N*D, cudaMemcpyDeviceToHost));
    if(d_out_coords_tangents) {
      gpuErrchk(cudaMemcpy(&coords_tangents[0], d_coords_tangents, sizeof(*d_coords_tangents)*N*D, cudaMemcpyDeviceToHost));      
    }

    gpuErrchk(cudaMemcpy(&params[0], d_params, sizeof(*d_params)*P, cudaMemcpyDeviceToHost));
    if(d_out_coords) {
      gpuErrchk(cudaMemcpy(&out_coords[0], d_out_coords, sizeof(*d_out_coords)*N*D, cudaMemcpyDeviceToHost));      
    }
    if(d_out_coords_tangents) {
      gpuErrchk(cudaMemcpy(&out_coords_tangents[0], d_out_coords_tangents, sizeof(*d_out_coords_tangents)*N*D, cudaMemcpyDeviceToHost));      
    }
    if(d_out_params_tangents) {
      gpuErrchk(cudaMemcpy(&out_params_tangents[0], d_out_params_tangents, sizeof(*d_out_params_tangents)*P, cudaMemcpyDeviceToHost));      
    }


    std::vector<double> born_radii(N, 0);
    std::vector<double> obc_chain(N, 0);
    std::vector<double> obc_chain_ri(N, 0);

    compute_born_radii<D>(
        coords,
        params,
        atomic_radii_idxs_,
        scale_factor_idxs_,
        dielectric_offset_,
        alpha_,
        beta_,
        gamma_,
        cutoff_,
        born_radii,
        obc_chain,
        obc_chain_ri
    );

    std::vector<double> out_forces(N*D, 0);

    // CPU
    std::vector<double> out_dU_dp(N*D, 0);

    compute_born_energy_and_forces<D>(
        coords,
        params,
        charge_param_idxs_,
        atomic_radii_idxs_,
        scale_factor_idxs_,
        born_radii,
        obc_chain,
        obc_chain_ri,
        alpha_,
        beta_,
        gamma_,
        dielectric_offset_,
        screening_,
        surface_tension_,
        solute_dielectric_,
        solvent_dielectric_,
        probe_radius_,
        cutoff_,
        out_forces,
        out_dU_dp
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    for(int i=0; i< out_forces.size(); i++) {
        out_coords[i] = static_cast<unsigned long long>((long long) (out_forces[i]*FIXED_EXPONENT));
    }

    if(d_out_coords) {
      gpuErrchk(cudaMemcpy(d_out_coords, &out_coords[0], sizeof(*d_out_coords)*N*D, cudaMemcpyHostToDevice));      
    }

    if(d_out_coords_tangents) {
      gpuErrchk(cudaMemcpy(d_out_coords_tangents, &out_coords_tangents[0], sizeof(*d_out_coords_tangents)*N*D, cudaMemcpyHostToDevice));      
    }

    if(d_out_params_tangents) {
      gpuErrchk(cudaMemcpy(d_out_params_tangents, &out_params_tangents[0], sizeof(*d_out_params_tangents)*P, cudaMemcpyHostToDevice));      
    }

    // gpuErrchk(cudaMemcpy(d_out_coords, &out_coords[0], N*D, cudaMemcpyHostToDevice));

}


template class GBSAReference<double, 4>;
template class GBSAReference<double, 3>;

template class GBSAReference<float, 4>;
template class GBSAReference<float, 3>;


}

