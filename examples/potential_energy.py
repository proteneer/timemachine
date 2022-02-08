from jax import grad
from jax import numpy as np
from jax import value_and_grad

from timemachine.fe.functional import construct_differentiable_interface
from timemachine.testsystems.relative import hif2a_ligand_pair

ff_params = hif2a_ligand_pair.ff.get_ordered_params()
unbound_potentials, sys_params, _, coords = hif2a_ligand_pair.prepare_vacuum_edge(ff_params)
U = construct_differentiable_interface(unbound_potentials)
box = np.eye(3) * 100
lam = 0.5

# can call U
U(coords, sys_params, box, lam)

# can call grad(U)
grad(U, argnums=(0, 1, 3))(coords, sys_params, box, lam)

# can call value_and_grad(U)
value_and_grad(U, argnums=(0, 1, 3))(coords, sys_params, box, lam)
