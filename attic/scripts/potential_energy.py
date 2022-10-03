from jax import grad
from jax import numpy as jnp
from jax import value_and_grad

from timemachine.fe.functional import construct_differentiable_interface
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology
from timemachine.fe.single_topology import SingleTopology
from timemachine.constants import DEFAULT_FF
from timemachine.ff import Forcefield
from timemachine.fe.utils import get_romol_conf

mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
forcefield = Forcefield.load_from_file(DEFAULT_FF)
st3 = SingleTopology(mol_a, mol_b, core, forcefield)
lam = 0.5
U = st3.get_U_fn(lam)
coords = st3.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

# can call U
U(coords)

# can call grad(U)
grad(U, argnums=(0,))(coords)

# can call value_and_grad(U)
value_and_grad(U, argnums=(0,))(coords)
