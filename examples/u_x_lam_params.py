from fe.functional import construct_differentiable_interface
from testsystems.relative import hif2a_ligand_pair
from jax import grad, value_and_grad

ff_params = hif2a_ligand_pair.ff
final_potentials, final_params, combined_masses, combined_coords = hif2a_ligand_pair.prepare_vacuum_edge(ff_params)
u_fxn = construct_differentiable_interface(sys_params=final_params, unbound_potentials=final_potentials)

# can call u_xn
u_fxn(x=combined_coords, lam=0.5, params=ff_params)

# can call grad(u_fxn)
grad(u_fxn, argnums=(0,1,2))(x=combined_coords, lam=0.5, params=ff_params)

# can call value_and_grad(u_fxn)
value_and_grad(u_fxn, argnums=(0,1,2))(x=combined_coords, lam=0.5, params=ff_params)
