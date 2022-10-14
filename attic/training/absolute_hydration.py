def set_up_smc_parameter_changes_at_endstates(
    mol,
    temperature=300.0,
    pressure=1.0,
    n_steps=1000,
    seed=2022,
    ff0=None,
    ff1=None,
    is_vacuum=False,
):
    """
    Prepare a system for using SMC to generate samples under
    different forcefields at each endstate.

    Parameters
    ----------
    is_vacuum: bool
        Set to True to set up the vacuum leg, using NVT.
        Set to False to set up the solvent leg, using NPT.

    Returns
    -------
    * reduced_potential_fxn
    * mover with lamb=None. The mover.lamb attribute must be set before use.
    """
    if type(seed) != int:
        seed = np.random.randint(1000)
        print(f"setting seed randomly to {seed}")
    else:
        print(f"setting seed to {seed}")

    np.random.seed(seed)

    # set up potentials
    ff0 = ff0 or Forcefield.load_from_file(DEFAULT_FF)
    ff1 = ff1 or Forcefield.load_from_file(DEFAULT_FF)

    if is_vacuum:
        potentials, params, masses, _ = enhanced.get_vacuum_phase_system_parameter_changes(mol, ff0, ff1)
    else:
        potentials, params, masses, _, _ = enhanced.get_solvent_phase_system_parameter_changes(mol, ff0, ff1)

    U_fn = functional.construct_differentiable_interface_fast(potentials, params)
    kBT = BOLTZ * temperature

    def reduced_potential_fxn(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam) / kBT

    for U, p in zip(potentials, params):
        U.bind(p)

    if is_vacuum:
        mover = moves.NVTMove(potentials, None, masses, temperature, n_steps, seed)
    else:
        mover = moves.NPTMove(potentials, None, masses, temperature, pressure, n_steps, seed)
    return reduced_potential_fxn, mover


def set_up_ahfe_system_for_smc_parameter_changes(
    mol,
    n_walkers,
    n_md_steps,
    resample_thresh,
    initial_samples,
    seed=2022,
    ff0=None,
    ff1=None,
    n_windows=10,
    is_vacuum=False,
):
    """
    Set up an absolute hydration free energy system such that
    the samples can be propagated using different forcefields
    at the end states.

    Parameters
    ----------
    mol:
        Molecule to use to generate samples.
    n_walkers: int
        Number of walkers to use
    n_md_steps: int
        Number of MD steps per walker
    resample_thresh: float
        Resample when the fraction ess falls below this value.
    initial_samples: Samples
        Initial set of unweighted samples generated using the initial forcefield.
    seed: int
        Random seed.
    ff0: Forcefield
        Initial forcefield (lam=0)
    ff1: Forcefield
        New forcefield (lam=1)
    n_windows: int
        Number of windows to use for parameter change.
    is_vacuum: bool
        True if this should be the vacuum leg or False for the solvent leg.

    Returns
    -------
        initial samples
        lambdas schedule
        propagate fxn
        log_prob fxn
        resample fxn
    """
    reduced_potential, mover = set_up_smc_parameter_changes_at_endstates(
        mol, n_steps=n_md_steps, seed=seed, ff0=ff0, ff1=ff1, is_vacuum=is_vacuum
    )
    np.random.seed(seed)

    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers, replace=True)
    samples = [initial_samples[i] for i in sample_inds]
    lambdas = np.linspace(0.0, 1.0, n_windows, endpoint=True)

    def propagate(xs, lam):
        mover.lamb = lam
        xs_next = [mover.move(x) for x in xs]
        return xs_next

    def log_prob(xs, lam):
        u_s = np.array([reduced_potential(x, lam) for x in xs])
        return -u_s

    resample = partial(smc.conditional_multinomial_resample, thresh=resample_thresh)

    return samples, lambdas, propagate, log_prob, resample


def generate_samples_for_smc_parameter_changes(
    mol: Chem.rdchem.Mol,
    ff0: Forcefield,
    ff1: Forcefield,
    initial_samples: smc.Samples,
    n_windows=10,
    is_vacuum=False,
    n_walkers=100,
    n_md_steps=100,
    seed=2022,
) -> smc.Samples:
    """
    Generate new samples under a modified potential.
    See `set_up_ahfe_system_for_smc_parameter_changes` for parameters.

    Returns
    -------
    New samples generated using ff1.
    """
    samples, lambdas, propagate, log_prob, resample = set_up_ahfe_system_for_smc_parameter_changes(
        mol,
        n_walkers=n_walkers,
        n_md_steps=n_md_steps,
        resample_thresh=0.6,
        initial_samples=initial_samples,
        ff0=ff0,
        ff1=ff1,
        is_vacuum=is_vacuum,
        n_windows=n_windows,
        seed=seed,
    )

    if is_vacuum:
        smc_result = smc.sequential_monte_carlo(initial_samples, lambdas, propagate, log_prob, resample)
        return smc.refine_samples(smc_result["traj"][-1], smc_result["log_weights_traj"][-1], propagate, lambdas[-1])
    else:
        smc_result = smc.sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)
        return smc.refine_samples(smc_result["traj"][-1], smc_result["log_weights_traj"][-1], propagate, lambdas[-1])
