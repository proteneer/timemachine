Code, documentation, experiments we want to retain for reference, but that we're not currently maintaining.

### Contents
* `docs/` -- write-up of initial vision for `timemachine`, involving efficient backpropagation through MD trajectories
* `jax_tricks` -- misc. Jax functions
* `modules` -- misc. modules
   * `reservoir_sampler.py`
   * `rmsd.py` -- compute rmsd under Jax
   * `potentials` -- deprecated potentials
     * `evp.py`
     * `pmi.py`
     * `gbsa.py` - GBSA implicit solvent model
     * `shape.py` - Calculate volume overlap between two molecules
   * `tests` -- deprecated tests
     * `test_shape.py` -- tests for potentials.shape
     * `dual_topology.py` -- test with TI and dual topology
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
