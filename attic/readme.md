Code, documentation, experiments we want to retain for reference, but that we're not currently maintaining.

### Contents
* `atom_mapping/` -- distance-based atom comparison for rdFMCS
* `docs/` -- write-up of initial vision for `timemachine`, involving efficient backpropagation through MD trajectories
* `docking/` -- Docking module that docks uses non-equilibrium switching
* `io/` -- Input/output utilities
* `jax_tricks` -- misc. Jax functions
* `modules` -- misc. modules
   * `reservoir_sampler.py`
   * `rmsd.py` -- compute rmsd under Jax
   * `potentials` -- deprecated potentials
     * `evp.py` -- eigenvalue problem solvers for small square matrices
     * `pmi.py` -- principal moments of inertia related code
     * `gbsa.py` -- GBSA implicit solvent model
     * `shape.py` -- Calculate volume overlap between two molecules
   * `tests` -- deprecated tests
     * `test_shape.py` -- tests for potentials.shape
     * `dual_topology.py` -- test with TI and dual topology
* `rabfe` -- Code previous used in RABFE
   * `free_energy.py` -- Free Energy classes for RABFE
   * `model_rabfe.py` -- Models for performing stages of RABFE
   * `test_rabfe_topology.py` -- Tests for rabfe topologies
   * `test_rabfe.py` -- Test RABFE
   * `topology.py` -- RABFE specific topology classes
   * `lambda_schedule.py` -- Lambda schedules specific to RABFE
   * `estimator.py` -- Estimating dGs for RABFE
   * `restraints.py` -- Restraints specific to RABFE
   * `test_restraints.py` -- Tests for restraints
   * `endpoint_correction.py` -- Endpoint correction for RABFE
   * `test_endpoint_correction.py` -- Tests for endpoint correction
* `optimize/` -- legacy protocol and forcefield optimization code
* `thermo_deriv/` -- numerical experiments with "thermodynamic derivative" estimators, adjusting LJ parameters to match observables
    * note: currently missing dependencies `thermo_deriv.lj_non_periodic.lennard_jones`, `thermo_deriv.lj.lennard_jones`.
    * note: `langevin_coefficients` dependency has since changed -- some scripts rely on a version of `langevin_coefficients` prior to PR #459
* `training/` -- classes and functions for an earlier training workflow
    * note: somewhat tailored to use TI estimates, constant volume simulations, GRPC, ...
* `scripts/`
  * `bootstrap_am1.py` -- script for approximating AM1BCC with atom types rather than bond types
  * `rhfe_single.py` runs three hydration free energy protocols that should give similar results:
    * taking the difference between two absolute hydration free energy calculations (each computed using TI and 4D decoupling)
    * computing a relative free energy using a complete atom-mapping (interpolating the valence energies, and interpolating the nonbonded parameters)
    * computing a relative free energy using a partial atom-mapping (the atom being deleted / inserted is handled using 4D decoupling)
  * `rbfe_single.py` -- script for running relative binding free energy with single topology
  * `estimator_variance.py` measures run-to-run variability as a function of number of MD steps performed per run
  * `generate_gradients.py` use symbolic differentiation for common functional forms to emit CUDA code
  * `npt_equilibration.py` samples the constant temperature and pressure ensemble for each of several lambda windows, by running thermostatted MD and pausing every few steps to apply a Monte Carlo barostat move
  * `ahfe.py` runs 4D decoupling simulations of aspirin in a waterbox, for later use in a TI estimator of the absolute hydration free energy.
  * `rhfe_dual.py` runs 4D decoupling simulations of aspirin and a version of aspirin that mutates an oxygen to a fluorine in a waterbox, subject to a centroid restraint, for later use in a TI estimator of the relative hydration free energy difference between these two compounds.
  * `hif2a/` runs calculations on several of the ligands in the hif2a benchmark dataset
    * `hif2a/generate_star_map.py` generates a "star map" (a single hub and several spokes), and serializes a collection of   RelativeFreeEnergy objects for (hub, spoke) pairs
    * `hif2a/fit_to_multiple_rbfes.py` fits nonbonded parameters to the experimental IC50s associated with these ligands
  * `overlap_test.py` tests two restraint types, one based on principal moments of inertia and one based on a measure of shape overlap.
  * `potential_energy.py` computes the potential energy of an "alchemical" system, as well as its derivatives w.r.t. coords, params, or lam
  * `run_smc_on_biphenyl.py` run smc on a biphenyl test system
  * `validate_relative_binding.py` run RABFE protocol
