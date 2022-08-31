import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.fe import topology
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.integrator import LangevinIntegrator
from timemachine.potentials import generic


def align_mols_by_core(
    mol_a: Chem.Mol, mol_b: Chem.Mol, core: NDArray, ff: Forcefield, n_steps: int = 100, k: float = 10000
) -> Tuple[NDArray, NDArray]:
    """
    Given two mols and a core mapping, simulate the mols in vacuum with harmonic bounds between them to get aligned poses.

    Returns the poses with the lowest RMSD between the cores.

    Parameters
    ----------
    mol_a: Chem.Mol
        First Ligand.

    mol_b: Chem.Mol
        Second Ligand.

    core: np.ndarray
        K x 2 mapping of coordinates in A to B. Assumes core[:, 0] matches mol_a and core[:, 1] matches mol_b

    ff: ff.Forcefield
        Wrapper class around a list of handlers.

    n_steps: int
        Number of steps to run vacuum simulation for.

    k: float
        Harmonic bond K value

    Returns
    -------
    tuple (mol_a_coords, mol_b_coords)
        New coords of the two molecules.


    """
    mol_a_conf = get_romol_conf(mol_a)
    mol_b_conf = get_romol_conf(mol_b)
    x0 = np.concatenate([mol_a_conf, mol_b_conf])
    masses = np.concatenate([get_mol_masses(mol_a), get_mol_masses(mol_b)])
    bond_indices = np.array(core)
    bond_indices[:, 1] += mol_a_conf.shape[0]
    top = topology.DualTopology(mol_a, mol_b, ff)

    vacuum_system = top.setup_end_state()
    U = vacuum_system.get_U_fn()

    fb = generic.HarmonicBond(bond_indices)
    restraint_params = np.zeros((len(core), 2))
    restraint_params[:, 0] = k
    bond_U = functools.partial(
        fb.to_reference(),
        params=jnp.array(restraint_params),
        box=None,
        lam=0.0,
    )

    def combined_U(x):
        return U(x) + bond_U(x)

    grad_fn = jax.jit(jax.grad(combined_U, argnums=(0)))

    def force_func(x):
        du_dx = grad_fn(x)
        return -du_dx

    # TBD 1e-3 step size is arbitrary
    intg = LangevinIntegrator(force_func, masses, 0.0, 1e-3, 0.0)
    key = jax.random.PRNGKey(2022)  # seed value doesn't matter with no friction and 0 temp
    xs, _ = intg.multiple_steps_lax(key, x0, np.zeros_like(x0), n_steps=n_steps)
    xs = np.array(xs)

    mol_a_core = xs[:, bond_indices[:, 0]]
    mol_b_core = xs[:, bond_indices[:, 1]]
    deltas = mol_a_core - mol_b_core
    rmsds = [np.linalg.norm(delta) for delta in deltas]
    best_alignment_idx = np.argmin(rmsds)

    conf_a = xs[best_alignment_idx][: mol_a.GetNumAtoms()]
    conf_b = xs[best_alignment_idx][mol_a.GetNumAtoms() :]
    return conf_a, conf_b
