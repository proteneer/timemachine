# This file contains utility functions to generate samples in the gas-phase.

import numpy as np
from fe import topology
from fe.utils import get_romol_conf
import jax
import jax.numpy as jnp

from md import builders, minimizer
from md.barostat.utils import get_group_indices, get_bond_list
from scipy.special import logsumexp
from fe import free_energy
from timemachine import lib
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients
from timemachine.potentials import bonded, nonbonded
from timemachine.constants import BOLTZ
import time

from jax import random as jrandom

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def _fori_steps(x0, v0, key0, grad_fn, num_steps, dt, ca, cbs, ccs):
    def body_fn(_, val):
        x_t, v_t, key = val
        du_dx = grad_fn(x_t)[0]
        v_mid = v_t + cbs * du_dx
        noise = jrandom.normal(key, v_t.shape)
        _, sub_key = jrandom.split(key)
        v_t = ca * v_mid + ccs * noise
        x_t += 0.5 * dt * (v_mid + v_t)
        return x_t, v_t, sub_key

    return jax.lax.fori_loop(0, num_steps, body_fn, (x0, v0, key0))


def _simulate(
    x0, temperature, U_fn, masses, steps_per_batch, num_batches, num_workers, seed=None
):
    dt = 1.5e-3
    friction = 1.0
    ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)
    cbs = np.expand_dims(cbs * -1, axis=-1)
    ccs = np.expand_dims(ccs, axis=-1)

    grad_fn = jax.jit(jax.grad(U_fn, argnums=(0,)))
    U_fn = jax.jit(U_fn)

    if seed is None:
        seed = int(time.time())

    @jax.jit
    def multiple_steps(x0, v0, key0):
        return _fori_steps(x0, v0, key0, grad_fn, steps_per_batch, dt, ca, cbs, ccs)

    v0 = np.zeros_like(x0)

    # jitting a pmap will result in a warning about inefficient data movement
    batched_multiple_steps_fn = jax.pmap(multiple_steps)

    xs_t = np.array([x0] * num_workers)
    vs_t = np.array([v0] * num_workers)
    keys_t = np.array([jrandom.PRNGKey(seed + idx) for idx in range(num_workers)])

    all_xs = []

    for batch_step in range(num_batches):
        #                                             [B,N,3][B,N,3][B,2]
        xs_t, vs_t, keys_t = batched_multiple_steps_fn(xs_t, vs_t, keys_t)
        all_xs.append(xs_t)

    # result has shape [num_workers, num_batches, num_atoms, num_dimensions]
    return np.transpose(np.array(all_xs), axes=[1, 0, 2, 3])


def identify_rotatable_bonds(mol):
    """
    Identify rotatable bonds in a molecule.

    Right now this is an extremely crude and inaccurate method that should *not* be used for production.
    This misses simple cases like benzoic acids, amides, etc. It also does not truncate out terminal bonds.

    Parameters
    ----------
    mol: ROMol
        Input molecule

    Returns
    -------
    set of 2-tuples
        Set of bonds identified as rotatable.

    """
    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern)
    # sanity check

    # print("identified", len(matches), "expected", rdMolDescriptors.CalcNumRotatableBonds(mol))
    assert len(matches) == rdMolDescriptors.CalcNumRotatableBonds(mol)

    sorted_matches = set()

    for i, j in matches:
        if j < i:
            i, j = j, i
        sorted_matches.add((i, j))

    return sorted_matches


class EnhancedState:
    def __init__(self, mol, ff):
        # parameterize the system
        self.mol = mol
        bt = topology.BaseTopology(mol, ff)
        self.bond_params, self.hb_potential = bt.parameterize_harmonic_bond(
            ff.hb_handle.params
        )
        self.angle_params, self.ha_potential = bt.parameterize_harmonic_angle(
            ff.ha_handle.params
        )
        self.proper_torsion_params, self.pt_potential = bt.parameterize_proper_torsion(
            ff.pt_handle.params
        )
        (
            self.improper_torsion_params,
            self.it_potential,
        ) = bt.parameterize_improper_torsion(ff.it_handle.params)
        self.nb_params, self.nb_potential = bt.parameterize_nonbonded(
            ff.q_handle.params, ff.lj_handle.params
        )

        self.box = None
        self.lamb = 0.0

    def _harmonic_bond_nrg(self, x):
        return bonded.harmonic_bond(
            x, self.bond_params, self.box, self.lamb, self.hb_potential.get_idxs()
        )

    def _harmonic_angle_nrg(self, x):
        return bonded.harmonic_angle(
            x, self.angle_params, self.box, self.lamb, self.ha_potential.get_idxs()
        )

    def _proper_torsion_nrg(self, x):
        return bonded.periodic_torsion(
            x,
            self.proper_torsion_params,
            self.box,
            self.lamb,
            self.pt_potential.get_idxs(),
        )

    def _improper_torsion_nrg(self, x):
        return bonded.periodic_torsion(
            x,
            self.improper_torsion_params,
            self.box,
            self.lamb,
            self.it_potential.get_idxs(),
        )

    def _nonbonded_nrg(self, x, decharge):
        exclusions = self.nb_potential.get_exclusion_idxs()
        scales = self.nb_potential.get_scale_factors()

        N = x.shape[0]
        charge_rescale_mask = np.ones((N, N))
        lj_rescale_mask = np.ones((N, N))

        if decharge:
            nb_params = jax.ops.index_update(self.nb_params, jax.ops.index[:, 0], 0)
        else:
            nb_params = self.nb_params

        for (i, j), (lj_scale, q_scale) in zip(exclusions, scales):
            charge_rescale_mask[i][j] = 1 - q_scale
            charge_rescale_mask[j][i] = 1 - q_scale
            lj_rescale_mask[i][j] = 1 - lj_scale
            lj_rescale_mask[j][i] = 1 - lj_scale

        beta = self.nb_potential.get_beta()
        cutoff = self.nb_potential.get_cutoff()
        lambda_plane_idxs = np.zeros(N)
        lambda_offset_idxs = np.zeros(N)

        # tbd: set to None
        box = np.eye(3) * 1000

        return nonbonded.nonbonded_v3(
            x,
            nb_params,
            box,
            self.lamb,
            charge_rescale_mask,
            lj_rescale_mask,
            beta,
            cutoff,
            lambda_plane_idxs,
            lambda_offset_idxs,
            runtime_validate=False,
        )

    def U_easy(self, x):
        """
        Gas-phase potential energy function typically used for the proposal distribution.
        This state has rotatable torsions fully turned off, and vdw radiis completely disabled.
        Note that this will may at times cross atropisomerism barriers in unphysical ways.

        Parameters
        ----------
        x: np.ndarray(N,3)
            Conformation of the input ligand

        Returns
        -------
        float
            Potential energy

        """
        # (ytz): torsion adjustments are disabled for now until we can make this more generalizable
        # ideas, for each bond, rotate by some small epsilon radians, if energy difference >
        # 10kJ/mol then we deem the bond as "non-rotatable"?

        easy_proper_torsion_idxs = []
        easy_proper_torsion_params = []

        # currently not optimal
        rotatable_bonds = identify_rotatable_bonds(self.mol)

        for idxs, params in zip(
            self.pt_potential.get_idxs(), self.proper_torsion_params
        ):
            _, j, k, _ = idxs
            if (j, k) in rotatable_bonds:
                print("turning off torsion", idxs)
                continue
            else:
                easy_proper_torsion_idxs.append(idxs)
                easy_proper_torsion_params.append(params)

        easy_proper_torsion_idxs = np.array(easy_proper_torsion_idxs, dtype=np.int32)
        easy_proper_torsion_params = np.array(
            easy_proper_torsion_params, dtype=np.float64
        )

        proper_torsion_nrg = bonded.periodic_torsion(
            x, easy_proper_torsion_params, self.box, self.lamb, easy_proper_torsion_idxs
        )

        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + proper_torsion_nrg
            + self._improper_torsion_nrg(x)
        )

    def U_full(self, x):
        """
        Fully interacting gas-phase potential energy.

        Parameters
        ----------
        x: np.ndarray(N,3)
            Conformation of the input ligand

        Returns
        -------
        float
            Potential energy

        """
        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + self._proper_torsion_nrg(x)
            + self._improper_torsion_nrg(x)
            + self._nonbonded_nrg(x, decharge=False)
        )

    def U_decharged(self, x):
        """
        Fully interacting, but decharged, gas-phase potential energy.

        Parameters
        ----------
        x: np.ndarray(N,3)
            Conformation of the input ligand

        Returns
        -------
        float
            Potential energy

        """
        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + self._proper_torsion_nrg(x)
            + self._improper_torsion_nrg(x)
            + self._nonbonded_nrg(x, decharge=True)
        )


def generate_samples(
    masses,
    x0,
    U_fn,
    temperature=300.0,
    steps_per_batch=1000,
    num_batches=1000,
    num_workers=None,
):
    """
    Generate samples from a chargeless gas-phase distribution where
    the torsional barriers for rotatable bonds have been significantly lowered.

    Parameters
    ----------
    mol: Chem.ROMol
        rdkit molecule

    n_steps: int
        number of steps

    sample_interval: int
        How often we sample

    num_workers: int
        How many jobs to run in parallel

    Returns
    -------
    tuple of np.ndarray with shapes [P,K,N,3], [K]
        Returns the frames and the associated energies, where P is the number
        of workers, K is the number of subsampled frames, N is the number of atoms

    """
    start_time = time.time()
    samples = _simulate(
        x0, temperature, U_fn, masses, steps_per_batch, num_batches, num_workers
    )

    print("duration:", time.time() - start_time)

    return samples


def generate_gas_phase_samples(
    mol, ff, temperature, U_target, steps_per_batch=250, num_batches=10000
):
    """
    Generate a set of gas-phase samples by running steps_per_batch * num_batches steps

    Parameters
    ----------
    mol: Chem.Mol

    ff: forcefield

    temperature: float

    U_target: fn
        Potential energy function we wish to re-weight into

    Returns
    -------
    3-tuple
        Return counts, samples, energies

    """
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    x0 = get_romol_conf(mol)

    kT = temperature * BOLTZ
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_workers = jax.device_count()

    state = EnhancedState(mol, ff)

    xs_easy = generate_samples(
        masses, x0, state.U_easy, temperature, steps_per_batch, num_batches, num_workers
    )

    num_atoms = mol.GetNumAtoms()

    # discard first few batches for burn-in and reshape into a single flat array
    xs_easy = xs_easy[:, 1000:, :, :]

    batch_U_easy_fn = jax.pmap(jax.vmap(state.U_easy))

    batch_U_target_fn = jax.pmap(jax.vmap(U_target))

    Us_target = batch_U_target_fn(xs_easy)
    Us_easy = batch_U_easy_fn(xs_easy)

    log_numerator = -Us_target.reshape(-1) / kT
    log_denominator = -Us_easy.reshape(-1) / kT

    log_weights = log_numerator - log_denominator
    weights = np.exp(log_weights - logsumexp(log_weights))

    # sample from weights
    sample_size = len(weights)
    idxs = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights)

    unique_target_kv = {}
    for i in idxs:
        if i not in unique_target_kv:
            unique_target_kv[i] = 0
        unique_target_kv[i] += 1

    # keys() and values() will always return in the same order in python3
    unique_target_idxs = np.array(list(unique_target_kv.keys()))
    unique_target_counts = np.array(list(unique_target_kv.values()))

    Us_target_unique = Us_target.reshape(-1)[unique_target_idxs]
    xs_target_unique = xs_easy.reshape(-1, num_atoms, 3)[unique_target_idxs]

    return unique_target_counts, xs_target_unique, Us_target_unique


def generate_solvent_phase_samples(
    mol, ff, temperature, steps_per_batch=250, num_batches=5000
):
    """
    Generate samples in the solvent phase.

    Parameters
    ----------
    mol: Chem.Mol

    ff: forcefield

    temperature: float

    U_target: fn
        Potential energy function we wish to re-weight into

    Returns
    -------
        xs, boxes, full_us, bps, nb_params, [water_topology, mol]

    """

    x0 = get_romol_conf(mol)

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    water_system, water_coords, water_box, water_topology = builders.build_water_system(
        3.0
    )
    water_box = (
        water_box + np.eye(3) * 0.5
    )  # add a small margin around the box for stability
    num_water_atoms = len(water_coords)
    afe = free_energy.AbsoluteFreeEnergy(mol, ff, decharge=False)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(
        ff_params, water_system, water_coords
    )

    dt = 1.5e-3
    friction = 1.0
    pressure = 1.0
    interval = 5

    box = water_box
    host_coords = coords[:num_water_atoms]
    new_host_coords = minimizer.minimize_host_4d(
        [mol], water_system, host_coords, ff, water_box
    )
    coords[:num_water_atoms] = new_host_coords

    bps = []
    for p, bp in zip(params, ubps):
        bps.append(bp.bind(p))

    all_impls = [bp.bound_impl(np.float32) for bp in bps]

    intg_equil = lib.LangevinIntegrator(temperature, 1e-4, friction, masses, 2021)
    intg_equil_impl = intg_equil.impl()

    # equilibration/minimization doesn't need a barostat
    equil_ctxt = custom_ops.Context(
        coords, np.zeros_like(coords), box, intg_equil_impl, all_impls, None
    )

    lamb = 0.0
    equil_schedule = np.ones(50000) * lamb
    equil_ctxt.multiple_steps(equil_schedule)

    x0 = equil_ctxt.get_x_t()
    v0 = np.zeros_like(x0)

    # production
    intg = lib.LangevinIntegrator(temperature, dt, friction, masses, 2021)
    intg_impl = intg.impl()

    # reset impls
    all_impls = [bp.bound_impl(np.float32) for bp in bps]

    bond_list = get_bond_list(ubps[0])
    group_idxs = get_group_indices(bond_list)

    barostat = lib.MonteCarloBarostat(
        len(masses), pressure, temperature, group_idxs, interval, 2022
    )
    barostat_impl = barostat.impl(all_impls)

    ctxt = custom_ops.Context(x0, v0, box, intg_impl, all_impls, barostat_impl)

    num_steps = steps_per_batch * num_batches

    lamb = 0.0
    lambda_windows = np.array([0.0])

    u_interval = steps_per_batch
    x_interval = steps_per_batch
    full_us, xs, boxes = ctxt.multiple_steps_U(
        lamb, num_steps, lambda_windows, u_interval, x_interval
    )

    # pdb_writer = PDBWriter([water_topology, mol], "solvent.pdb")
    # pdb_writer.write_frame(xs[0]*10)
    # pdb_writer.write_frame(xs[-1]*10)
    # pdb_writer.close()

    burn_in = int(0.85 * num_batches)

    return (
        xs[burn_in:],
        boxes[burn_in:],
        full_us[burn_in:],
        bps[-1],
        params[-1],
        [water_topology, mol],
    )
