import numpy as np

from timemachine.ff import Forcefield
from timemachine.ff.handlers.bonded import (
    HarmonicAngleHandler,
    HarmonicBondHandler,
    ImproperTorsionHandler,
    ProperTorsionHandler,
)
from timemachine.ff.handlers.nonbonded import (
    LennardJonesHandler,
    LennardJonesIntraHandler,
    SimpleChargeHandler,
    SimpleChargeIntraHandler,
)

# bonded
placeholder_hb_handle = HarmonicBondHandler(smirks=["[*:1]~[*:2]"], params=np.array([[1e5, 1e-1]]), props=None)
placeholder_ha_handle = HarmonicAngleHandler(
    smirks=["[*:1]~[*:2]~[*:3]"], params=np.array([[1e2, np.pi / 2]]), props=None
)
placeholder_pt_handle = ProperTorsionHandler(
    smirks=["[*:1]~[*:2]~[*:3]~[*:4]"], params=np.array([[1.0, 0.0, 1]]), props=None
)
placeholder_it_handle = ImproperTorsionHandler(
    smirks=["[*:1]~[#6X3,#7X3:2](~[*:3])~[*:4]"], params=np.array([[1.0, np.pi, 2]]), props=None
)

# charge / chargeintra / chargesolvent
_q_smirks = ["[*:1]"]
_q_params = np.zeros(1)

placeholder_q_handle = SimpleChargeHandler(smirks=_q_smirks, params=_q_params, props=None)
placeholder_q_handle_intra = SimpleChargeIntraHandler(smirks=_q_smirks, params=_q_params, props=None)

# lj / ljintra / ljsolvent
_lj_smirks = ["[*:1]"]
_lj_params = np.array([[0.1, 1.0]])

placeholder_lj_handle = LennardJonesHandler(smirks=_lj_smirks, params=_lj_params, props=None)
placeholder_lj_handle_intra = LennardJonesIntraHandler(smirks=_lj_smirks, params=_lj_params, props=None)

# construct
placeholder_ff = Forcefield(
    hb_handle=placeholder_hb_handle,
    ha_handle=placeholder_ha_handle,
    pt_handle=placeholder_pt_handle,
    it_handle=placeholder_it_handle,
    q_handle=placeholder_q_handle,
    q_handle_intra=placeholder_q_handle_intra,
    lj_handle=placeholder_lj_handle,
    lj_handle_intra=placeholder_lj_handle_intra,
    protein_ff="amber99sbildn",
    water_ff="amber14/tip3p",
)

# serialize
with open("params/placeholder_ff.py", "w") as f:
    f.write(placeholder_ff.serialize())
