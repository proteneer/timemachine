import jax
import jax.numpy as jnp
import numpy as np

from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.potentials import SummedPotential
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def test_jax_differentiable_interface():
    """Assert that the computation of U and its derivatives using the
    C++ code path produces equivalent results to doing the
    summation in Python"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    vac_sys = st.setup_intermediate_state(0.5)
    x_a = get_romol_conf(st.mol_a)
    x_b = get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    box = np.eye(3) * 100

    bps = vac_sys.get_U_fns()
    potentials = [bp.potential for bp in bps]
    sys_params = [np.array(bp.params) for bp in bps]

    for precision in [np.float32, np.float64]:

        gpu_impls = [p.to_gpu(precision) for p in potentials]

        def U_ref(coords, sys_params, box):
            return jnp.sum(jnp.array([f(coords, params, box) for f, params in zip(gpu_impls, sys_params)]))

        U = SummedPotential(potentials, sys_params).to_gpu(precision).call_with_params_list
        args = (coords, sys_params, box)
        np.testing.assert_array_equal(U(*args), U_ref(*args))

        argnums = (0, 1)
        dU_dx_ref, dU_dps_ref = jax.grad(U_ref, argnums)(*args)
        dU_dx, dU_dps = jax.grad(U, argnums)(*args)

        np.testing.assert_allclose(dU_dx, dU_dx_ref)

        assert len(dU_dps) == len(dU_dps_ref)
        for dU_dp, dU_dp_ref in zip(dU_dps, dU_dps_ref):
            np.testing.assert_allclose(dU_dp, dU_dp_ref)
