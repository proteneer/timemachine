from jax import config

config.update("jax_enable_x64", True)
import numpy as np
from jax import jit

from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md.builders import build_water_system

ff = Forcefield.load_default()


def make_water_cluster_potentials():
    """get potentials for an n=2 water cluster in effectively open boundary conditions"""

    system, positions, _, _ = build_water_system(
        0.50, water_ff=ff.water_ff  # very small box width --> just a couple waters
    )
    num_waters = len(positions) // 3
    assert num_waters == 2

    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    open_boundary_conditions_box = 100.0 * np.eye(3)

    @jit
    def U_ref(x):
        return sum(bp(x, open_boundary_conditions_box) for bp in bps)

    gpu_bps = []
    for bp in bps:
        gpu_bps.append(bp.to_gpu(np.float32))

    def U_gpu(x):
        return sum(gpu_bp(x, open_boundary_conditions_box) for gpu_bp in gpu_bps)

    return U_ref, U_gpu


def test_coincident_opposite_charges():
    U_ref, U_gpu = make_water_cluster_potentials()

    # pair of waters near each other: should be highly unfavorable. if LJ eps == 0, can spuriously appear favorable.
    x0 = np.array(
        [
            [-0.26525125, 0.90447467, -0.13290655],
            [-0.0301779, 0.63181125, 0.07063769],
            [-0.029876, 0.63356334, 0.0699187],
            [-0.03056342, 0.63168464, 0.068646],
            [-0.26512399, 0.90563125, -0.13317576],
            [-0.00299566, 0.55669363, -0.00323221],
        ]
    )

    # minimized version of x0, should be even more spurious / close to -inf, if LJ  eps == 0.
    x1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.23468019182676886, -0.27501762075242653, 0.20330731275297725],
            [0.2348330960655381, -0.2742542244781153, 0.20311446995569982],
            [0.23468022555779056, -0.2750175955775044, 0.20330733589134378],
            [-0.000466390775090475, -0.0041670468725262255, 0.0009657274517013703],
            [0.2619482156903966, -0.3504305740722373, 0.13028281682847767],
        ]
    )

    for x in [x0, x1]:
        for U_fxn in [U_ref, U_gpu]:
            U = U_fxn(x)
            assert (U > 0) or np.isnan(U)
