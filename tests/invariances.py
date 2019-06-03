from jax.test_util import check_grads
from scipy.stats import special_ortho_group
import numpy as np
import jax

def assert_potential_invariance(energy_fn, x0, params, box=None):
    # note: all potentials must support non-periodic implementations,
    # defined by box being None

    # explicitly check without box even if box is not None
    check_grads(energy_fn, (x0, params, None), order=1, eps=1e-5)
    check_grads(energy_fn, (x0, params, None), order=2, eps=1e-7)

    # check with box if present
    if box is not None:
        check_grads(energy_fn, (x0, params, box), order=1, eps=1e-5)
        check_grads(energy_fn, (x0, params, box), order=2, eps=1e-7)

    # test translational and rotational invariance of
    # energy and its derivatives, with and without box
    energy = energy_fn(x0,params,None)
    force_fn = jax.grad(energy_fn,argnums=(0,))
    forces = force_fn(x0,params,None)[0]
    dEdp = jax.jacfwd(energy_fn,argnums=(1,))
    d2Edxdp = jax.jacfwd(dEdp,argnums=(0,))
    dparam = dEdp(x0,params,None)[0]
    mixed = d2Edxdp(x0,params,None)[0][0]

    for _ in range(3):
        trans_vector = np.random.rand(3)
        trans_x = x0 + trans_vector
        trans_energy = energy_fn(trans_x,params,None)
        trans_forces = force_fn(trans_x,params,None)[0]
        trans_dEdp = dEdp(trans_x,params,None)[0]
        trans_mixed = d2Edxdp(trans_x,params,None)[0][0]
        np.testing.assert_almost_equal(trans_energy,energy,decimal=5)
        np.testing.assert_array_almost_equal(trans_forces,forces,decimal=5)
        np.testing.assert_array_almost_equal(trans_dEdp,dparam,decimal=5)
        np.testing.assert_array_almost_equal(trans_mixed,mixed,decimal=5)
         
    for _ in range(3):
        rot_matrix = special_ortho_group.rvs(3)
        rot_x = np.matmul(x0, rot_matrix)
        rot_energy = energy_fn(rot_x,params,None)
        rot_forces = force_fn(rot_x,params,None)[0]
        rot_dEdp = dEdp(rot_x,params,None)[0]
        rot_mixed = d2Edxdp(rot_x,params,None)[0][0]
        np.testing.assert_array_almost_equal(rot_energy,energy,decimal=5)
        np.testing.assert_array_almost_equal(rot_forces,np.matmul(forces,rot_matrix),decimal=5) 
        np.testing.assert_array_almost_equal(rot_dEdp,dparam,decimal=5)
        for i in range(rot_mixed.shape[0]):
            np.testing.assert_array_almost_equal(rot_mixed[i],np.matmul(mixed[i],rot_matrix),decimal=5) 

    for _ in range(3):
        trans_vector = np.random.rand(3)
        rot_matrix = special_ortho_group.rvs(3)
        comp_x = np.matmul(x0,rot_matrix) + trans_vector
        comp_energy = energy_fn(comp_x,params,None)
        comp_forces = force_fn(comp_x,params,None)[0]
        comp_dEdp = dEdp(comp_x,params,None)[0]
        comp_mixed = d2Edxdp(comp_x,params,None)[0][0]
        np.testing.assert_array_almost_equal(comp_energy,energy,decimal=5)
        np.testing.assert_array_almost_equal(comp_forces,np.matmul(forces,rot_matrix),decimal=5) 
        np.testing.assert_array_almost_equal(comp_dEdp,dparam,decimal=5)
        for i in range(comp_mixed.shape[0]):
            np.testing.assert_array_almost_equal(comp_mixed[i],np.matmul(mixed[i],rot_matrix),decimal=5) 
        
    if box is not None:
        energy = energy_fn(x0,params,box)
        force_fn = jax.grad(energy_fn,argnums=(0,))
        forces = force_fn(x0,params,box)[0]
        dEdp = jax.jacfwd(energy_fn,argnums=(1,))
        d2Edxdp = jax.jacfwd(dEdp,argnums=(0,))
        dparam = dEdp(x0,params,box)[0]
        mixed = d2Edxdp(x0,params,box)[0][0]

        for _ in range(3):
            trans_vector = np.random.rand(3)
            trans_x = x0 + trans_vector
            trans_energy = energy_fn(trans_x,params,box)
            trans_forces = force_fn(trans_x,params,box)[0]
            trans_dEdp = dEdp(trans_x,params,box)[0]
            trans_mixed = d2Edxdp(trans_x,params,box)[0][0]
            np.testing.assert_almost_equal(trans_energy, energy,decimal=5)
            np.testing.assert_array_almost_equal(trans_forces, forces,decimal=5)
            np.testing.assert_array_almost_equal(trans_dEdp,dparam,decimal=5)
            np.testing.assert_array_almost_equal(trans_mixed, mixed,decimal=5)
        
#         for _ in range(3):,decimal=5
#             rot_matrix = special_ortho_group.rvs(3)
#             rot_x = np.matmul(x0, rot_matrix)
#             rot_energy = energy_fn(rot_x,params,box)
#             rot_forces = force_fn(rot_x,params,box)[0]
#             rot_mixed = d2Edxdp(rot_x,params,box)[0][0]
#             np.testing.assert_almost_equal(rot_energy,energy,decimal=5)
#             np.testing.assert_array_almost_equal(rot_forces,np.matmul(forces,rot_matrix),decimal=5)
#             np.testing.assert_array_almost_equal(rot_dEdp,dparam,decimal=5)
#             for i in range(rot_mixed.shape[0]):
#                 np.testing.assert_array_almost_equal(rot_mixed[i],np.matmul(mixed[i],rot_matrix),decimal=5) 
            
#         for _ in range(3):
#             trans_vector = np.random.rand(3)
#             rot_matrix = special_ortho_group.rvs(3)
#             comp_x = np.matmul(x0,rot_matrix) + trans_vector
#             comp_energy = energy_fn(comp_x,params,box)
#             comp_forces = force_fn(comp_x,params,box)[0]
#             comp_mixed = d2Edxdp(trans_x,params,box)[0][0]
#             np.testing.assert_almost_equal(comp_energy,energy,decimal=5)
#             np.testing.assert_array_almost_equal(comp_forces,np.matmul(forces,rot_matrix),decimal=5)
#             np.testing.assert_array_almost_equal(rot_dEdp,dparam,decimal=5)
#             for i in range(comp_mixed.shape[0]):
#                 np.testing.assert_array_almost_equal(comp_mixed[i],np.matmul(mixed[i],rot_matrix),decimal=5) 
