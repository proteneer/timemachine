* `abfe.py`, `config.ini` -- script for estimating absolute binding free energy
* `bootstrap.py` -- functions for bootstrapping TI estimates
* `hydration_fe.py`, `hydration_cfg.ini` -- script for training to hydration free energies
  * note: also contains a function `recenter(conf, box)`
* `hydration_model.py` -- defines a `simulate` function that computes hydration free energies and an estimate of parameter gradients
* `hydration_setup.py` -- functions for combining coordinates, potentials, and vjps
* `simulation.py` -- named tuple for x, v, potentials, integrator
* `trainer.py` -- defines `Trainer` and `Timer` classes, `Trainer::run_mol` computes an absolute binding free energy, parameter gradients, and updates force field parameters
