Measure and plot some performance characteristics of local versions of HMC on the DHFR benchmark system.

* `generate_equilibrium_samples.py` runs 1 million steps of NPT using production settings (Langevin(dt=2.5fs, gamma=1/ps), MCBarostat(interval=15)) and saves snapshots every 1000 steps.
* `polish_samples.py` "polishes" the snapshots generated from Langevin by applying 100 steps of Barker MCMC to each snapshot. This doesn't have much apparent effect, but reduces the number of things to worry about in subsequent steps. (Can be more confident that we're starting from equilibrium-looking samples.)
* `hmc.py` defines `LocalHMC` and `GlobalHMC` moves (these are intended to be consolidated and added to timemachine.md.moves shortly)
* `profile_hmc.py` -- measures performance characteristics by simulating individual HMC proposals from randomly selected equilibrium samples. saves `n_atoms_moved`, `proposal_norm`, `log_accept_prob`, and `timing` for each proposal. Varies step size and local MD radius on grids, with fixed n_steps and fixed local restraint k. Each proposal is centered on a randomly selected protein atom.
* `plot_hmc.py` -- plots average acceptance probability as a function of step size for global HMC and several local HMC variants.
