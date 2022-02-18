Code, documentation, experiments we want to retain for reference, but that we're not currently maintaining.

### Contents
* `docs/` -- write-up of initial vision for `timemachine`, involving efficient backpropagation through MD trajectories
* `jax_tricks` -- misc. Jax functions
* `thermo_deriv/` -- numerical experiments with "thermodynamic derivative" estimators, adjusting LJ parameters to match observables
    * note: currently missing dependencies `thermo_deriv.lj_non_periodic.lennard_jones`, `thermo_deriv.lj.lennard_jones`.
    * note: `langevin_coefficients` dependency has since changed -- some scripts rely on a version of `langevin_coefficients` prior to PR #459
* `training/` -- classes and functions for an earlier training workflow
    * note: somewhat tailored to use TI estimates, constant volume simulations, GRPC, ...
* `scripts/`
  * `bootstrap_am1.py` -- script for approximating AM1BCC with atom types rather than bond types
