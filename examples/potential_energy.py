from jax import grad
from jax import numpy as np
from jax import value_and_grad

from timemachine.fe.functional import construct_differentiable_interface
from timemachine.testsystems.relative import hif2a_ligand_pair

rfe = hif2a_ligand_pair
unbound_potentials, sys_params, _ = rfe.prepare_host_edge(rfe.ff.get_ordered_params())
coords = rfe.prepare_combined_coords()
U = construct_differentiable_interface(unbound_potentials)
box = np.eye(3) * 100
lam = 0.5

# can call U
U(coords, sys_params, box, lam)

# can call grad(U)
grad(U, argnums=(0, 1, 3))(coords, sys_params, box, lam)

# can call value_and_grad(U)
value_and_grad(U, argnums=(0, 1, 3))(coords, sys_params, box, lam)
