#include <stdexcept>
#include <iostream> 
#include "fixed_point.hpp"
#include "gbsa.hpp"
#include "kernel_utils.cuh"

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
    int N = numberOfAtoms;

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

             if (offsetRadiusI < rScaledRadiusJ) {
                double rInverse = 1.0/r;

                double l_ij = 0;
                if(offsetRadiusI > abs(r - scaledRadiusJ)) {
                  l_ij = offsetRadiusI;
                } else {
                  l_ij = abs(r - scaledRadiusJ);
                }

                // double l_ij     = offsetRadiusI > abs(r - scaledRadiusJ) ? offsetRadiusI : abs(r - scaledRadiusJ);
                l_ij     = 1.0/l_ij;

                double u_ij     = 1.0/rScaledRadiusJ;

                double l_ij2    = l_ij*l_ij;
                double u_ij2    = u_ij*u_ij;
 
                double ratio    = log((u_ij/l_ij));
                double term     = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);


                printf("term1 %f\n", term*0.5*offsetRadiusI);
                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details

                // if (offsetRadiusI < (scaledRadiusJ - r)) {
                   // term += 2.0*(radiusIInverse - l_ij);
                // }
                sum += term;

               }
            }
         }

       // std::cout << "psi " << i_idx << " " << sum/2 << std::endl; 
       // OBC-specific code (Eqs. 6-8 in paper)
       // std::cout << "ori" << offsetRadiusI << std::endl;

       sum              *= 0.5*offsetRadiusI;
       // std::cout << "sum " << sum << std::endl;
       double sum2       = sum*sum;
       double sum3       = sum*sum2;
       double tanhSum    = tanh(alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3);
 
       born_radii[i_idx]      = 1.0/(1.0/offsetRadiusI - tanhSum/radiusI); 


       // dRi/dPsi
       obc_chain[i_idx]       = offsetRadiusI*(alpha_obc - 2.0*beta_obc*sum + 3.0*gamma_obc*sum2);
       obc_chain[i_idx]       = (1.0 - tanhSum*tanhSum)*obc_chain[i_idx]/radiusI;
       obc_chain[i_idx]      *= born_radii[i_idx]*born_radii[i_idx];
       
       // flip the signs later
       // dRi/dri
       obc_chain_ri[i_idx]    = -1.0/(offsetRadiusI*offsetRadiusI) + tanhSum/(radiusI*radiusI);
       obc_chain_ri[i_idx]   *= -born_radii[i_idx]*born_radii[i_idx];

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
    std::vector<double> &out_forces
) {

    // constants

    // const int numberOfAtoms = _obcParameters->getNumberOfAtoms();
    const int numberOfAtoms = atomic_radii_idxs.size();
    const int N = numberOfAtoms;

    const double dielectricOffset = dielectric_offset;
    const double cutoffDistance = cutoff;
    const double soluteDielectric = solute_dielectric;
    const double solventDielectric = solvent_dielectric;
    double preFactor;
    if (soluteDielectric != 0.0 && solventDielectric != 0.0) {
          // preFactor = 2.0*electric_constant*((1.0/soluteDielectric) - (1.0/solventDielectric));
        preFactor = -screening*((1.0/soluteDielectric) - (1.0/solventDielectric));    
        // printf("prefactor %f\n", preFactor);
    } else {
        preFactor = 0.0;
    }

    // ---------------------------------------------------------------------------------------

    // compute Born radii
    // vector<double> born_radii(numberOfAtoms);
    // computeBornRadii(atomCoordinates, born_radii);

    // set energy/forces to zero

    double obcEnergy = 0.0;
    std::vector<double> bornForces(numberOfAtoms, 0.0);

    // ---------------------------------------------------------------------------------------

    // compute the nonpolar solvation via ACE approximation
    // (ytz): disabled for now
    // if (includeAceApproximation()) {
       // computeAceNonPolarForce(_obcParameters, born_radii, &obcEnergy, bornForces);
    // }

    // const double probeRadius          = obcParameters->getProbeRadius();
    // const double surface_tension    = obcParameters->getPi4Asolv();

    // const vector<double>& atomicRadii   = obcParameters->getAtomicRadii();
    // int numberOfAtoms                     = obcParameters->getNumberOfAtoms();

    // the original ACE equation is based on Eq.2 of

    // M. Schaefer, C. Bartels and M. Karplus, "Solution Conformations
    // and Thermodynamics of Structured Peptides: Molecular Dynamics
    // Simulation with an Implicit Solvation Model", J. Mol. Biol.,
    // 284, 835-848 (1998)  (ACE Method)

    // The original equation includes the factor (atomicRadii[atomI]/bornRadii[atomI]) to the first power,
    // whereas here the ratio is raised to the sixth power: (atomicRadii[atomI]/bornRadii[atomI])**6

    // This modification was made by Jay Ponder who observed it gave better correlations w/
    // observed values. He did not think it was important enough to write up, so there is
    // no paper to cite.



    // for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
    //     if (born_radii[atomI] > 0.0) {
    //         double atomic_radii = params[atomic_radii_idxs[atomI]];
    //         double r            = atomic_radii + probe_radius;
    //         double ratio6       = pow(atomic_radii/born_radii[atomI], 6.0);
    //         double saTerm       = surface_tension*r*r*ratio6;
    //         obcEnergy          += saTerm;
    //         bornForces[atomI]  -= 6.0*saTerm/born_radii[atomI]; 
    //     }
    // }
 
    // ---------------------------------------------------------------------------------------

    // first main loop

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
 
       double partialChargeI = preFactor*params[charge_param_idxs[atomI]];
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
          double Gpol               = (partialChargeI*partialChargeJ)/denominator; 
          double dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  

          double dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;
          
          double energy = Gpol;

          if (atomI != atomJ) {

              // TBD: determine what we should do with cutoff
                // energy -= partialChargeI*partialCharges[atomJ]/cutoff;
              bornForces[atomJ]        += dGpol_dalpha2_ij*born_radii[atomI];

              for(int d=0; d < D; d++) {
                out_forces[atomI*D+d] += dxs[d]*dGpol_dr;
                out_forces[atomJ*D+d] -= dxs[d]*dGpol_dr;
              }

          } else {
             energy *= 0.5;
          }

          obcEnergy         += energy;
          bornForces[atomI] += dGpol_dalpha2_ij*born_radii[atomJ];

       }
    }

    // ---------------------------------------------------------------------------------------

    // second main loop

    // compute factor that depends only on the outer loop index (this is just the outer derivative
    // of the born radius function)
    // for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
    //    bornForces[atomI] *= born_radii[atomI]*born_radii[atomI]*obc_chain[atomI];      
    // }

    // obc chain is derivative of the born radius with respect to the psi sum

    // can we factor out LHS?

    std::vector<double> atomic_radii_derivatives(N, 0);


    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
      // order matters here
      // atomic_radii_derivatives[atomI] = bornForces[atomI] * obc_chain_ri[atomI]; // do obc chain separately
      bornForces[atomI] *= obc_chain[atomI];
    }

    std::vector<double> dPsi_dx(N*D, 0);


    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
 
       // radius w/ dielectric offset applied

       double radiusI        = params[atomic_radii_idxs[atomI]];
       double offsetRadiusI  = radiusI - dielectricOffset;

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

             double scaleFactorJ       = params[scale_factor_idxs[atomJ]];
             double scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
             double scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
             double rScaledRadiusJ     = r + scaledRadiusJ;

             // this is *not* correct
             // dL/dr & dU/dr are zero (this can be shown analytically)
             // removed from calculation

             if (offsetRadiusI < rScaledRadiusJ) {

                // double l_ij          = offsetRadiusI > abs(r - scaledRadiusJ) ? offsetRadiusI : abs(r - scaledRadiusJ);
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
                if(offsetRadiusI > abs(r - scaledRadiusJ)) {
                  double term = (1.0/2.0)*(dielectricOffset - radiusI)*(-1.0/4.0*r*(pow(r - scaleFactorJ*(dielectricOffset - radiusJ), -2) - 1/pow(dielectricOffset - radiusI, 2)) + 1.0/(r - scaleFactorJ*(dielectricOffset - radiusJ)) + 1.0/(dielectricOffset - radiusI) + (1.0/4.0)*pow(scaleFactorJ, 2)*pow(dielectricOffset - radiusJ, 2)*(pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), -2) - 1/pow(-dielectricOffset + radiusI, 2))/r - 1.0/2.0*log(-(dielectricOffset - radiusI)/(r - scaleFactorJ*(dielectricOffset - radiusJ)))/r);

                  // printf("a term2 %f\n", term);
                  // de = (-1.0/2.0*dielectricOffset + (1.0/2.0)*radiusI)*(-1.0/2.0*r/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3) + (5.0/4.0)/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) - 1.0/4.0/pow(-dielectricOffset + radiusI, 2) + (1.0/2.0)*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)/(r*pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3)) - 1.0/2.0/(r*(r + scaleFactorJ*(-dielectricOffset + radiusJ))) - 1.0/4.0*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)*(-1/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) + pow(-dielectricOffset + radiusI, -2))/pow(r, 2) - 1.0/2.0*log((-dielectricOffset + radiusI)/(r + scaleFactorJ*(-dielectricOffset + radiusJ)))/pow(r, 2));

                  de = -1.0/2.0*r/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3) + (5.0/4.0)/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) - 1.0/4.0/pow(-dielectricOffset + radiusI, 2) + (1.0/2.0)*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)/(r*pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3)) - 1.0/2.0/(r*(r + scaleFactorJ*(-dielectricOffset + radiusJ))) - 1.0/4.0*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)*(-1/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) + pow(-dielectricOffset + radiusI, -2))/pow(r, 2) - 1.0/2.0*log((-dielectricOffset + radiusI)/(r + scaleFactorJ*(-dielectricOffset + radiusJ)))/pow(r, 2);

                  printf("%d de a RHS: %.8f\n", atomI, de);

                } else {
                  double term = -1.0/2.0*(dielectricOffset - radiusI)*(-1.0/4.0*r*(pow(r + scaleFactorJ*(dielectricOffset - radiusJ), -2) - 1/pow(r - scaleFactorJ*(dielectricOffset - radiusJ), 2)) + 1.0/fabs(r + scaleFactorJ*(dielectricOffset - radiusJ)) - 1/(r - scaleFactorJ*(dielectricOffset - radiusJ)) - 1.0/4.0*pow(scaleFactorJ, 2)*pow(dielectricOffset - radiusJ, 2)*(-1/pow(r + scaleFactorJ*(dielectricOffset - radiusJ), 2) + pow(r - scaleFactorJ*(dielectricOffset - radiusJ), -2))/r + (1.0/2.0)*log(fabs(r + scaleFactorJ*(dielectricOffset - radiusJ))/(r - scaleFactorJ*(dielectricOffset - radiusJ)))/r);
                  // printf("b term2 %f\n", term);

                  // de = (-1.0/2.0*dielectricOffset + (1.0/2.0)*radiusI)*((1.0/4.0)*r*(-2/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3) + 2/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 3)) + (5.0/4.0)/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) - (((r - scaleFactorJ*(-dielectricOffset + radiusJ)) > 0) - ((r - scaleFactorJ*(-dielectricOffset + radiusJ)) < 0))/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 2) - 1.0/4.0/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 2) + (1.0/4.0)*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)*(2/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3) - 2/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 3))/r + (1.0/2.0)*(r + scaleFactorJ*(-dielectricOffset + radiusJ))*((((r - scaleFactorJ*(-dielectricOffset + radiusJ)) > 0) - ((r - scaleFactorJ*(-dielectricOffset + radiusJ)) < 0))/(r + scaleFactorJ*(-dielectricOffset + radiusJ)) - fabs(r - scaleFactorJ*(-dielectricOffset + radiusJ))/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2))/(r*fabs(r - scaleFactorJ*(-dielectricOffset + radiusJ))) - 1.0/4.0*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)*(-1/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) + pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), -2))/pow(r, 2) - 1.0/2.0*log(fabs(r - scaleFactorJ*(-dielectricOffset + radiusJ))/(r + scaleFactorJ*(-dielectricOffset + radiusJ)))/pow(r, 2));
                  de = (1.0/4.0)*r*(-2/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3) + 2/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 3)) + (5.0/4.0)/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) - (((r - scaleFactorJ*(-dielectricOffset + radiusJ)) > 0) - ((r - scaleFactorJ*(-dielectricOffset + radiusJ)) < 0))/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 2) - 1.0/4.0/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 2) + (1.0/4.0)*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)*(2/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 3) - 2/pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), 3))/r + (1.0/2.0)*(r + scaleFactorJ*(-dielectricOffset + radiusJ))*((((r - scaleFactorJ*(-dielectricOffset + radiusJ)) > 0) - ((r - scaleFactorJ*(-dielectricOffset + radiusJ)) < 0))/(r + scaleFactorJ*(-dielectricOffset + radiusJ)) - fabs(r - scaleFactorJ*(-dielectricOffset + radiusJ))/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2))/(r*fabs(r - scaleFactorJ*(-dielectricOffset + radiusJ))) - 1.0/4.0*pow(scaleFactorJ, 2)*pow(-dielectricOffset + radiusJ, 2)*(-1/pow(r + scaleFactorJ*(-dielectricOffset + radiusJ), 2) + pow(r - scaleFactorJ*(-dielectricOffset + radiusJ), -2))/pow(r, 2) - 1.0/2.0*log(fabs(r - scaleFactorJ*(-dielectricOffset + radiusJ))/(r + scaleFactorJ*(-dielectricOffset + radiusJ)))/pow(r, 2);

                    printf("%d de b RHS: %.8f\n", atomI, de);
                }

                de = 0.5*bornForces[atomI]*de/r;

                for(int d=0; d < D; d++) {
                    // 1/d is for the derivative of the xyz itself
                    dPsi_dx[atomI*D+d] += dxs[d]*de;
                    dPsi_dx[atomJ*D+d] -= dxs[d]*de;
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


    std::cout << "energy" << obcEnergy << std::endl;
    // return obcEnergy;
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
      std::cout << "NOT null" << std::endl;
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
        out_forces
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

