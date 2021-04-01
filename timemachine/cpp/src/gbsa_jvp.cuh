#include <stdexcept>
#include <iostream> 
#include "fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"

namespace jankmachine {


template <int D>
double reduce_born_forces_jvp(
    const std::vector<Surreal<double> >& coords,
    const std::vector<double>& params,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<Surreal<double> >& born_radii,
    const std::vector<Surreal<double> >& obc_chain,
    const std::vector<Surreal<double> >& obc_chain_ri,
    const double surface_tension, // surface area factor
    const double probe_radius,
    std::vector<Surreal<double> > &bornForces, // dU/Ri
    std::vector<double> &out_MvP
) {

    // constants
    const int numberOfAtoms = atomic_radii_idxs.size();
    const int N = numberOfAtoms;


    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {

        Surreal<double> radii_derivs(0,0);
        if (born_radii[atomI].real > 0.0) {
            double atomic_radii = params[atomic_radii_idxs[atomI]];
            double r            = atomic_radii + probe_radius;
            Surreal<double> ar           = atomic_radii/born_radii[atomI];
            Surreal<double> ar2          = ar*ar;
            Surreal<double> ar4          = ar2*ar2;
            Surreal<double> ratio6          = ar4*ar2;
            Surreal<double> saTerm       = surface_tension*r*r*ratio6;
            bornForces[atomI]  -= 6.0*saTerm/born_radii[atomI]; 
            Surreal<double> br2 = born_radii[atomI]*born_radii[atomI];
            Surreal<double> br4 = br2*br2;
            Surreal<double> br6 = br4*br2;
            radii_derivs += 2*pow(atomic_radii, 5)*surface_tension*(probe_radius + atomic_radii)*(3*probe_radius + 4*atomic_radii)/br6;
        }
        radii_derivs += bornForces[atomI] * obc_chain_ri[atomI];
        out_MvP[atomic_radii_idxs[atomI]] += radii_derivs.imag;
        bornForces[atomI] *= obc_chain[atomI];
    }

}


// ported over from OpenMM with minor corrections
template<int D>
void compute_born_radii_jvp(
    const std::vector<Surreal<double> >& coords,
    const std::vector<double>& params,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<int>& scale_factor_idxs,
    const double dielectric_offset,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    const double cutoff,
    std::vector<Surreal<double> > & born_radii,
    std::vector<Surreal<double> > & obc_chain,
    std::vector<Surreal<double> > & obc_chain_ri) {

    int numberOfAtoms = atomic_radii_idxs.size();
    int N = numberOfAtoms;

    if(coords.size() / D != numberOfAtoms) {
        throw std::runtime_error("compute born radii number of atoms are inconsistent");
    }

    for (int i_idx = 0; i_idx < numberOfAtoms; i_idx++) {
      
       double radiusI         = params[atomic_radii_idxs[i_idx]];
       double offsetRadiusI   = radiusI - dielectric_offset;
       double radiusIInverse  = 1.0/offsetRadiusI;
       Surreal<double> sum(0, 0);

       // HCT code
       for (int j_idx = 0; j_idx < numberOfAtoms; j_idx++) {

          if (j_idx != i_idx) {

             Surreal<double> r(0, 0);
             for(int d=0; d < D; d++) {
                Surreal<double> dx = coords[i_idx*D+d] - coords[j_idx*D+d];
                r += dx*dx;
             }
             r = sqrt(r);

             double offsetRadiusJ   = params[atomic_radii_idxs[j_idx]] - dielectric_offset; 
             double scaledRadiusJ   = offsetRadiusJ*params[scale_factor_idxs[j_idx]];
             Surreal<double> rScaledRadiusJ  = r + scaledRadiusJ;
             Surreal<double> rSubScaledRadiusJ =  r - scaledRadiusJ;
             if (offsetRadiusI < rScaledRadiusJ.real) {
                Surreal<double> rInverse = 1.0/r;

                Surreal<double> l_ij(0, 0);
                if(offsetRadiusI > abs(rSubScaledRadiusJ).real) {
                  l_ij.real = offsetRadiusI;
                  l_ij.imag = 0;
                } else {
                  l_ij = abs(rSubScaledRadiusJ);
                }

                l_ij     = 1.0/l_ij;

                Surreal<double> u_ij     = 1.0/rScaledRadiusJ;

                Surreal<double> l_ij2    = l_ij*l_ij;
                Surreal<double> u_ij2    = u_ij*u_ij;
 
                Surreal<double> ratio    = log((u_ij/l_ij));
                Surreal<double> term     = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r).real) {
                   term += 2.0*(radiusIInverse - l_ij);
                }
                sum += term;

               }
            }
         }
       sum              *= 0.5*offsetRadiusI;
       Surreal<double> sum2       = sum*sum;
       Surreal<double> sum3       = sum*sum2;
       Surreal<double> tanhSum    = tanh(alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3);
 
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



template<int D>
double compute_born_first_loop_jvp(
    const std::vector<Surreal<double> >& coords,
    const std::vector<double>& params,
    const std::vector<int>& charge_param_idxs,
    const std::vector<Surreal<double> >& born_radii,
    const double prefactor,
    const double cutoff,
    std::vector<Surreal<double> > &bornForces,
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
    std::vector<Surreal<double> > charge_derivs(N, Surreal<double>(0, 0));

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
 
        double partialChargeI = params[charge_param_idxs[atomI]];
        for (int atomJ = atomI; atomJ < numberOfAtoms; atomJ++) {

            Surreal<double> r2(0,0);
            Surreal<double> dxs[D] = {Surreal<double>(0, 0)};
            for(int d=0; d < D; d++) {
                Surreal<double> dx = coords[atomI*D+d] - coords[atomJ*D+d];
                dxs[d] = dx;
                r2 += dx*dx;
            }
            Surreal<double> r = sqrt(r2);
            Surreal<double> alpha2_ij          = born_radii[atomI]*born_radii[atomJ];
            Surreal<double> D_ij               = r2/(4.0*alpha2_ij);

            Surreal<double> expTerm            = exp(-D_ij);
            Surreal<double> denominator2       = r2 + alpha2_ij*expTerm; 
            Surreal<double> denominator        = sqrt(denominator2);

            double partialChargeJ     = params[charge_param_idxs[atomJ]];
            Surreal<double> Gpol               = (prefactor*partialChargeI*partialChargeJ)/denominator; 

            Surreal<double> dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  
            Surreal<double> dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;

            // printf("%d %d dGpol_dalpha2_ij %f\n", atomI, atomJ, dGpol_dalpha2_ij);

            Surreal<double> energy = Gpol;

            Surreal<double> dE_dqi = prefactor*partialChargeJ/denominator;
            Surreal<double> dE_dqj = prefactor*partialChargeI/denominator;

            if (atomI != atomJ) {

                // TBD: determine what we should do with cutoff
                // energy -= partialChargeI*partialCharges[atomJ]/cutoff;
                bornForces[atomJ]        += dGpol_dalpha2_ij*born_radii[atomI];
                for(int d=0; d < D; d++) {
                    out_forces[atomI*D+d] += (dxs[d]*dGpol_dr).imag;
                    out_forces[atomJ*D+d] -= (dxs[d]*dGpol_dr).imag;
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
        dU_dp[charge_param_idxs[i]] += charge_derivs[i].imag;
    }

};

template <int D>
double compute_born_energy_and_forces_jvp(
    const std::vector<Surreal<double> >& coords,
    const std::vector<double>& params,
    const std::vector<int>& atomic_radii_idxs,
    const std::vector<int>& scale_factor_idxs,
    const std::vector<Surreal<double> > & born_radii,
    const std::vector<Surreal<double> > & obc_chain,
    const std::vector<Surreal<double> > & obc_chain_ri,
    const double dielectric_offset,
    // const double screening,
    // const double surface_tension, // surface area factor
    // const double solute_dielectric,
    // const double solvent_dielectric,
    // const double probe_radius,
    const double cutoff,
    std::vector<Surreal<double> > &bornForces,
    std::vector<double> &out_HvP,
    std::vector<double> &out_MvP
) {

    // constants
    const int numberOfAtoms = atomic_radii_idxs.size();
    const int N = numberOfAtoms;

    const double dielectricOffset = dielectric_offset;
    const double cutoffDistance = cutoff;

    std::vector<Surreal<double> > dPsi_dx(N*D, Surreal<double>(0, 0));
    std::vector<Surreal<double> > dPsi_dri(N, Surreal<double>(0, 0));
    std::vector<Surreal<double> > dPsi_dsi(N, Surreal<double>(0, 0));

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
 
       // radius w/ dielectric offset applied

       double radiusI        = params[atomic_radii_idxs[atomI]];
       double offsetRadiusI  = radiusI - dielectricOffset;
       double offsetRadiusI2 = offsetRadiusI*offsetRadiusI;
       double offsetRadiusI3 = offsetRadiusI2*offsetRadiusI;

       for (int atomJ = 0; atomJ < numberOfAtoms; atomJ++) {

          if (atomJ != atomI) {

             Surreal<double> r2(0, 0);
             Surreal<double> dxs[D] = {Surreal<double>(0,0)};
             for(int d=0; d < D; d++) {
                Surreal<double> dx = coords[atomI*D+d] - coords[atomJ*D+d];
                dxs[d] = dx;
                r2 += dx*dx;
             }
             Surreal<double> r = sqrt(r2);

             // radius w/ dielectric offset applied

             double radiusJ            = params[atomic_radii_idxs[atomJ]];
             double offsetRadiusJ      = radiusJ - dielectricOffset;
             double offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

             double scaleFactorJ       = params[scale_factor_idxs[atomJ]];
             double scaleFactorJ2      = scaleFactorJ*scaleFactorJ;
             double scaleFactorJ3      = scaleFactorJ2*scaleFactorJ;
             double scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
             double scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
             Surreal<double> rScaledRadiusJ     = r + scaledRadiusJ;
             Surreal<double> rScaledRadiusJ2    = rScaledRadiusJ*rScaledRadiusJ;
             Surreal<double> rScaledRadiusJ3    = rScaledRadiusJ2*rScaledRadiusJ;

             // dL/dr & dU/dr are zero (this can be shown analytically)
             // removed from calculation

             if (offsetRadiusI < rScaledRadiusJ.real) {

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
                Surreal<double> de(0, 0); // derivative of Psi wrt the distance
                Surreal<double> dpsi_dri(0, 0);
                Surreal<double> dpsi_drj(0, 0);
                Surreal<double> dpsi_dsj(0, 0);

                Surreal<double> rSubScaledRadiusJ = r - scaledRadiusJ;
                Surreal<double> rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                Surreal<double> rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                // factor out as much as we can to outside of the conditional for reduce convergence
                if(offsetRadiusI > abs(rSubScaledRadiusJ).real) {
                  Surreal<double> term = 0.5*(-offsetRadiusI)*(-0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + 1.0/rScaledRadiusJ + 1.0/(-offsetRadiusI) + 0.25*scaleFactorJ2*offsetRadiusJ2*(1/rScaledRadiusJ2 - 1/offsetRadiusI2)/r - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r);
                  de = -0.5*r/rScaledRadiusJ3 + (5.0/4.0)/rScaledRadiusJ2 - 0.25/offsetRadiusI2 + 0.5*scaleFactorJ2*offsetRadiusJ2/(r*rScaledRadiusJ3) - 0.5/(r*rScaledRadiusJ) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r2 - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r2;
                  dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + offsetRadiusI*(0.5*r/offsetRadiusI3 - 1/offsetRadiusI2 - 0.5*scaleFactorJ2*offsetRadiusJ2/(r*offsetRadiusI3) + 0.5/(r*offsetRadiusI)) - 1/rScaledRadiusJ + 1.0/offsetRadiusI + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*log(offsetRadiusI/rScaledRadiusJ)/r;
                  dpsi_drj = offsetRadiusI*(-0.5*r*scaleFactorJ/rScaledRadiusJ3 + scaleFactorJ/rScaledRadiusJ2 + 0.5*scaleFactorJ3*offsetRadiusJ2/(r*rScaledRadiusJ3) + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r - 0.5*scaleFactorJ/(r*rScaledRadiusJ));
                  dpsi_dsj = offsetRadiusI*(0.25*r*(2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3 + offsetRadiusJ/rScaledRadiusJ2 - 0.25*scaleFactorJ2*offsetRadiusJ2*(2*dielectricOffset - 2*radiusJ)/(r*rScaledRadiusJ3) + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*(-offsetRadiusJ)/(r*rScaledRadiusJ));
                  if(offsetRadiusI < (scaledRadiusJ - r).real) {
                   de += 0;
                   dpsi_dri += 0;
                   dpsi_drj += 0;
                   dpsi_dsj += 0;
                  }

                } else {
                  Surreal<double> term = -0.5*(-offsetRadiusI)*(-0.25*r*(1/rSubScaledRadiusJ2 - 1/rScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rSubScaledRadiusJ2 + 1/rScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r);
                  de = 0.25*r*(-2/rScaledRadiusJ3 + 2/rSubScaledRadiusJ3) + (5.0/4.0)/rScaledRadiusJ2 - sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 - 0.25/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*(2/rScaledRadiusJ3 - 2/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(sign(rSubScaledRadiusJ)/rScaledRadiusJ - fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r2 - 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r2;
                  dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/rSubScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r;
                  dpsi_drj = offsetRadiusI*(0.25*r*(-2*scaleFactorJ/rScaledRadiusJ3 - 2*scaleFactorJ/rSubScaledRadiusJ3) + scaleFactorJ/rScaledRadiusJ2 + scaleFactorJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.25*scaleFactorJ2*offsetRadiusJ2*(2*scaleFactorJ/rScaledRadiusJ3 + 2*scaleFactorJ/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(-scaleFactorJ*sign(rSubScaledRadiusJ)/rScaledRadiusJ - scaleFactorJ*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                  dpsi_dsj = offsetRadiusI*(0.25*r*(-(-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 + (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3) + offsetRadiusJ/rScaledRadiusJ2 + offsetRadiusJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*((-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 - (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3)/r + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*rScaledRadiusJ*((-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rScaledRadiusJ + (-offsetRadiusJ)*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                  if (offsetRadiusI < (scaledRadiusJ - r).real) {
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


    // std::vector<double> &out_HvP,
    // std::vector<double> &out_MvP
    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
      for(int d=0; d < D; d++) {
        out_HvP[atomI*D+d] += dPsi_dx[atomI*D+d].imag;
      }
    }

    // for(int i=0; i < dPsi_dri.size(); i++) {
    //   std::cout << "dPsi_dri: " << dPsi_dri[i]+atomic_radii_derivatives[i] << std::endl;
    // }

    for(int i=0; i < dPsi_dri.size(); i++) {
      // std::cout << "dPsi_dri parts: " << dPsi_dri[i] << " " << atomic_radii_derivatives[i] << std::endl;
      // out_MvP[atomic_radii_idxs[i]] += dPsi_dri[i].imag + atomic_radii_derivatives[i].imag;
      out_MvP[atomic_radii_idxs[i]] += dPsi_dri[i].imag;
    }

    for(int i=0; i < dPsi_dsi.size(); i++) {
      out_MvP[scale_factor_idxs[i]] += dPsi_dsi[i].imag;
    }

    // for(int i=0; i < charge_derivs.size(); i++) {
    //   // std::cout << "???" << charge_derivs[i] << std::endl;
    //   out_MvP[charge_param_idxs[i]] += charge_derivs[i].imag;
    // }

    // std::cout << "energy" << obcEnergy.real << std::endl;
    // return obcEnergy;
}


}

