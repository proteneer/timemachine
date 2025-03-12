import numpy as np
from rdkit import Chem

from timemachine.fe import model_utils
from timemachine.fe.free_energy import InitialState, get_context
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator
from timemachine.potentials.bonded import kahan_angle


def test_nitrile_stability():
    # test that running a nitrile simulation is numerically stable
    # and that the average angle is greater than 3 rads.
    mol = Chem.MolFromMolBlock(
        """
  Mrv2311 03142420142D

  3  2  0  0  0  0            999 V2000
   -0.4939    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6150    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  3  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    #     mol = Chem.MolFromMolBlock(
    #         """
    #   Mrv2311 03142420322D

    #   3  2  0  0  0  0            999 V2000
    #    -0.1604    0.4921    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -0.0070   -0.0100    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    #     0.5174    0.0128    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #   1  2  1  0  0  0  0
    #   2  3  1  0  0  0  0
    # M  END
    # $$$$""",
    #         removeHs=False,
    #     )

    ff = Forcefield.load_default()
    top = BaseTopology(mol, ff)

    vs = top.setup_end_state()
    bps = vs.get_U_fns()
    print("bps", bps)
    masses = [a.GetMass() for a in mol.GetAtoms()]
    masses = model_utils.apply_hmr(masses, [[0, 1], [1, 2]])
    intg = LangevinIntegrator(300.0, 1.0e-3, 1.0, masses, seed=2024)
    baro = None
    x0 = get_romol_conf(mol)
    v0 = np.zeros_like(x0)
    box = np.eye(3) * 10
    lamb = 0.0
    ligand_idxs = np.array([0, 1, 2], dtype=np.int32)
    protein_idxs = np.array([], dtype=np.int32)
    init_state = InitialState(bps, intg, baro, x0, v0, box, lamb, ligand_idxs, protein_idxs)

    # (ytz): leave for plotting
    # import matplotlib.pyplot as plt
    # x_vals = np.linspace(2.2, np.pi, 100)
    # k0, a0 = bps[-1].params[0]
    # from timemachine.constants import DEFAULT_KT
    # def q_fn(x):
    #     return np.exp(-1 / DEFAULT_KT * (k0 / 2 * (x - a0) ** 2))
    # from scipy.integrate import quad
    # Z = quad(q_fn, 0, np.pi)[0]
    # def p_fn(x):
    #     return q_fn(x) / Z
    # y_vals = [p_fn(x) for x in x_vals]
    # plt.plot(x_vals, y_vals)

    ctxt = get_context(init_state)
    ctxt.multiple_steps(10_000)  # burn_in
    xs, _ = ctxt.multiple_steps(n_steps=100_000, store_x_interval=1000)
    angles = []
    for x in xs:
        angles.append(kahan_angle(x[0], x[1], x[2], 0.0))

    assert np.mean(angles) > 3.0
    assert np.amax(np.abs(xs)) < 15.0  # no coordinates blew-up
