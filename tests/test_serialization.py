import numpy as np
import pytest

from timemachine.ff.handlers import bonded, nonbonded
from timemachine.ff.handlers.deserialize import deserialize_handlers
from timemachine.ff.handlers.serialize import bin_to_str

pytestmark = [pytest.mark.nocuda]


def test_harmonic_bond():
    patterns = [
        ["[#6X4:1]-[#6X4:2]", 0.1, 0.2],
        ["[#6X4:1]-[#6X3:2]", 99.0, 99.0],
        ["[#6X4:1]-[#6X3:2]=[#8X1+0]", 99.0, 99.0],
        ["[#6X3:1]-[#6X3:2]", 99.0, 99.0],
        ["[#6X3:1]:[#6X3:2]", 99.0, 99.0],
        ["[#6X3:1]=[#6X3:2]", 99.0, 99.0],
        ["[#6:1]-[#7:2]", 0.1, 0.2],
        ["[#6X3:1]-[#7X3:2]", 99.0, 99.0],
        ["[#6X4:1]-[#7X3:2]-[#6X3]=[#8X1+0]", 99.0, 99.0],
        ["[#6X3:1](=[#8X1+0])-[#7X3:2]", 99.0, 99.0],
        ["[#6X3:1]-[#7X2:2]", 99.0, 99.0],
        ["[#6X3:1]:[#7X2,#7X3+1:2]", 99.0, 99.0],
        ["[#6X3:1]=[#7X2,#7X3+1:2]", 99.0, 99.0],
        ["[#6:1]-[#8:2]", 99.0, 99.0],
        ["[#6X3:1]-[#8X1-1:2]", 99.0, 99.0],
        ["[#6X4:1]-[#8X2H0:2]", 0.3, 0.4],
        ["[#6X3:1]-[#8X2:2]", 99.0, 99.0],
        ["[#6X3:1]-[#8X2H1:2]", 99.0, 99.0],
        ["[#6X3a:1]-[#8X2H0:2]", 99.0, 99.0],
        ["[#6X3:1](=[#8X1])-[#8X2H0:2]", 99.0, 99.0],
        ["[#6:1]=[#8X1+0,#8X2+1:2]", 99.0, 99.0],
        ["[#6X3:1](~[#8X1])~[#8X1:2]", 99.0, 99.0],
        ["[#6X3:1]~[#8X2+1:2]~[#6X3]", 99.0, 99.0],
        ["[#6X2:1]-[#6:2]", 99.0, 99.0],
        ["[#6X2:1]-[#6X4:2]", 99.0, 99.0],
        ["[#6X2:1]=[#6X3:2]", 99.0, 99.0],
        ["[#6:1]#[#7:2]", 99.0, 99.0],
        ["[#6X2:1]#[#6X2:2]", 99.0, 99.0],
        ["[#6X2:1]-[#8X2:2]", 99.0, 99.0],
        ["[#6X2:1]-[#7:2]", 99.0, 99.0],
        ["[#6X2:1]=[#7:2]", 99.0, 99.0],
        ["[#16:1]=[#6:2]", 99.0, 99.0],
        ["[#6X2:1]=[#16:2]", 99.0, 99.0],
        ["[#7:1]-[#7:2]", 99.0, 99.0],
        ["[#7X3:1]-[#7X2:2]", 99.0, 99.0],
        ["[#7X2:1]-[#7X2:2]", 99.0, 99.0],
        ["[#7:1]:[#7:2]", 99.0, 99.0],
        ["[#7:1]=[#7:2]", 99.0, 99.0],
        ["[#7+1:1]=[#7-1:2]", 99.0, 99.0],
        ["[#7:1]#[#7:2]", 99.0, 99.0],
        ["[#7:1]-[#8X2:2]", 99.0, 99.0],
        ["[#7:1]~[#8X1:2]", 99.0, 99.0],
        ["[#8X2:1]-[#8X2:2]", 99.0, 99.0],
        ["[#16:1]-[#6:2]", 99.0, 99.0],
        ["[#16:1]-[#1:2]", 99.0, 99.0],
        ["[#16:1]-[#16:2]", 99.0, 99.0],
        ["[#16:1]-[#9:2]", 99.0, 99.0],
        ["[#16:1]-[#17:2]", 99.0, 99.0],
        ["[#16:1]-[#35:2]", 99.0, 99.0],
        ["[#16:1]-[#53:2]", 99.0, 99.0],
        ["[#16X2,#16X1-1,#16X3+1:1]-[#6X4:2]", 99.0, 99.0],
        ["[#16X2,#16X1-1,#16X3+1:1]-[#6X3:2]", 99.0, 99.0],
        ["[#16X2:1]-[#7:2]", 99.0, 99.0],
        ["[#16X2:1]-[#8X2:2]", 99.0, 99.0],
        ["[#16X2:1]=[#8X1,#7X2:2]", 99.0, 99.0],
        ["[#16X4,#16X3!+1:1]-[#6:2]", 99.0, 99.0],
        ["[#16X4,#16X3:1]~[#7:2]", 99.0, 99.0],
        ["[#16X4,#16X3:1]-[#8X2:2]", 99.0, 99.0],
        ["[#16X4,#16X3:1]~[#8X1:2]", 99.0, 99.0],
        ["[#15:1]-[#1:2]", 99.0, 99.0],
        ["[#15:1]~[#6:2]", 99.0, 99.0],
        ["[#15:1]-[#7:2]", 99.0, 99.0],
        ["[#15:1]=[#7:2]", 99.0, 99.0],
        ["[#15:1]~[#8X2:2]", 99.0, 99.0],
        ["[#15:1]~[#8X1:2]", 99.0, 99.0],
        ["[#16:1]-[#15:2]", 99.0, 99.0],
        ["[#15:1]=[#16X1:2]", 99.0, 99.0],
        ["[#6:1]-[#9:2]", 99.0, 99.0],
        ["[#6X4:1]-[#9:2]", 0.6, 0.7],
        ["[#6:1]-[#17:2]", 99.0, 99.0],
        ["[#6X4:1]-[#17:2]", 99.0, 99.0],
        ["[#6:1]-[#35:2]", 99.0, 99.0],
        ["[#6X4:1]-[#35:2]", 99.0, 99.0],
        ["[#6:1]-[#53:2]", 99.0, 99.0],
        ["[#6X4:1]-[#53:2]", 99.0, 99.0],
        ["[#7:1]-[#9:2]", 99.0, 99.0],
        ["[#7:1]-[#17:2]", 99.0, 99.0],
        ["[#7:1]-[#35:2]", 99.0, 99.0],
        ["[#7:1]-[#53:2]", 99.0, 99.0],
        ["[#15:1]-[#9:2]", 99.0, 99.0],
        ["[#15:1]-[#17:2]", 99.0, 99.0],
        ["[#15:1]-[#35:2]", 99.0, 99.0],
        ["[#15:1]-[#53:2]", 99.0, 99.0],
        ["[#6X4:1]-[#1:2]", 99.0, 99.0],
        ["[#6X3:1]-[#1:2]", 99.0, 99.0],
        ["[#6X2:1]-[#1:2]", 99.0, 99.0],
        ["[#7:1]-[#1:2]", 99.0, 99.0],
        ["[#8:1]-[#1:2]", 99.0, 99.1],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    hbh = bonded.HarmonicBondHandler(smirks, params, None)

    obj = hbh.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    assert len(all_handlers) == 1

    new_hbh = all_handlers[0]
    np.testing.assert_equal(new_hbh.smirks, hbh.smirks)
    np.testing.assert_equal(new_hbh.params, hbh.params)

    assert new_hbh.props == hbh.props


def test_proper_torsion():
    # proper torsions have a variadic number of terms

    patterns = [
        ["[*:1]-[#6X3:2]=[#6X3:3]-[*:4]", [[99.0, 99.0, 99.0]]],
        ["[*:1]-[#6X3:2]=[#6X3:3]-[#35:4]", [[99.0, 99.0, 99.0]]],
        ["[#9:1]-[#6X3:2]=[#6X3:3]-[#35:4]", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        ["[#35:1]-[#6X3:2]=[#6X3:3]-[#35:4]", [[7.0, 8.0, 9.0], [1.0, 3.0, 5.0], [4.0, 4.0, 4.0]]],
        ["[#9:1]-[#6X3:2]=[#6X3:3]-[#9:4]", [[7.0, 8.0, 9.0]]],
    ]

    smirks = [x[0] for x in patterns]
    params = [x[1] for x in patterns]

    ph = bonded.ProperTorsionHandler(smirks, params, None)
    obj = ph.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    assert len(all_handlers) == 1

    new_ph = all_handlers[0]
    np.testing.assert_equal(new_ph.smirks, ph.smirks)
    np.testing.assert_equal(new_ph.params, ph.params)
    assert new_ph.props == ph.props


def test_improper_torsion():
    patterns = [
        ["[*:1]~[#6X3:2](~[*:3])~[*:4]", 1.5341333333333333, 3.141592653589793, 2.0],
        ["[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#7X3$(*~[#6X3]):2](~[*:3])~[*:4]", 1.3946666666666667, 3.141592653589793, 2.0],
        ["[*:1]~[#7X3$(*~[#7X2]):2](~[*:3])~[*:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#7X3$(*@1-[*]=,:[*][*]=,:[*]@1):2](~[*:3])~[*:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#6X3:2](=[#7X2,#7X3+1:3])~[#7:4]", 99.0, 99.0, 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2], x[3]] for x in patterns])
    imph = bonded.ImproperTorsionHandler(smirks, params, None)

    obj = imph.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    assert len(all_handlers) == 1

    new_imph = all_handlers[0]
    np.testing.assert_equal(new_imph.smirks, imph.smirks)
    np.testing.assert_equal(new_imph.params, imph.params)
    assert new_imph.props == imph.props


def test_simple_charge_handler():
    patterns = [
        ["[#1:1]", 99.0],
        ["[#1:1]-[#6X4]", 99.0],
        ["[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X4]~[*+1,*+2]", 99.0],
        ["[#1:1]-[#6X3]", 99.0],
        ["[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X2]", 99.0],
        ["[#1:1]-[#7]", 99.0],
        ["[#1:1]-[#8]", 99.0],
        ["[#1:1]-[#16]", 99.0],
        ["[#6:1]", 0.7],
        ["[#6X2:1]", 99.0],
        ["[#6X4:1]", 0.1],
        ["[#8:1]", 99.0],
        ["[#8X2H0+0:1]", 0.5],
        ["[#8X2H1+0:1]", 99.0],
        ["[#7:1]", 0.3],
        ["[#16:1]", 99.0],
        ["[#15:1]", 99.0],
        ["[#9:1]", 1.0],
        ["[#17:1]", 99.0],
        ["[#35:1]", 99.0],
        ["[#53:1]", 99.0],
        ["[#3+1:1]", 99.0],
        ["[#11+1:1]", 99.0],
        ["[#19+1:1]", 99.0],
        ["[#37+1:1]", 99.0],
        ["[#55+1:1]", 99.0],
        ["[#9X0-1:1]", 99.0],
        ["[#17X0-1:1]", 99.0],
        ["[#35X0-1:1]", 99.0],
        ["[#53X0-1:1]", 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([x[1] for x in patterns])
    props = None

    sch = nonbonded.SimpleChargeHandler(smirks, params, props)
    obj = sch.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    assert len(all_handlers) == 1

    new_sch = all_handlers[0]
    np.testing.assert_equal(new_sch.smirks, sch.smirks)
    np.testing.assert_equal(new_sch.params, sch.params)
    assert new_sch.props == sch.props


def test_gbsa_handler():
    patterns = [
        ["[*:1]", 99.0, 99.0],
        ["[#1:1]", 99.0, 99.0],
        ["[#1:1]~[#7]", 99.0, 99.0],
        ["[#6:1]", 0.1, 0.2],
        ["[#7:1]", 0.3, 0.4],
        ["[#8:1]", 0.5, 0.6],
        ["[#9:1]", 0.7, 0.8],
        ["[#14:1]", 99.0, 99.0],
        ["[#15:1]", 99.0, 99.0],
        ["[#16:1]", 99.0, 99.0],
        ["[#17:1]", 99.0, 99.0],
    ]

    props = {
        "solvent_dielectric": 78.3,  # matches OBC2,
        "solute_dielectric": 1.0,
        "probe_radius": 0.14,
        "surface_tension": 28.3919551,
        "dielectric_offset": 0.009,
        # GBOBC1
        "alpha": 0.8,
        "beta": 0.0,
        "gamma": 2.909125,
    }

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])

    gbh = nonbonded.GBSAHandler(smirks, params, props)

    obj = gbh.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    assert len(all_handlers) == 1

    new_gbh = all_handlers[0]
    np.testing.assert_equal(new_gbh.smirks, gbh.smirks)
    np.testing.assert_equal(new_gbh.params, gbh.params)
    assert new_gbh.props == gbh.props


def test_am1bcc():
    smirks = []
    params = []
    props = None

    am1 = nonbonded.AM1BCCHandler(smirks, params, props)
    obj = am1.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    am1 = all_handlers[0]
    np.testing.assert_equal(am1.smirks, am1.smirks)
    np.testing.assert_equal(am1.params, am1.params)
    assert am1.props == am1.props


def test_resp():
    smirks = []
    params = []
    props = None

    am1 = nonbonded.RESPHandler(smirks, params, props)
    obj = am1.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    am1 = all_handlers[0]
    np.testing.assert_equal(am1.smirks, am1.smirks)
    np.testing.assert_equal(am1.params, am1.params)
    assert am1.props == am1.props


def test_am1ccc():
    patterns = [
        ["[#6X4:1]-[#1:2]", 0.46323257920556493],
        ["[#6X3$(*=[#8,#16]):1]-[#6a:2]", 0.24281402370571598],
        ["[#6X3$(*=[#8,#16]):1]-[#8X1,#8X2:2]", 1.0620166764992722],
        ["[#6X3$(*=[#8,#16]):1]=[#8X1$(*=[#6X3]-[#8X2]):2]", 2.227759732057297],
        ["[#6X3$(*=[#8,#16]):1]=[#8X1,#8X2:2]", 2.8182928673804217],
        ["[#6a:1]-[#8X1,#8X2:2]", 0.5315976926761063],
        ["[#6a:1]-[#1:2]", 0.0],
        ["[#6a:1]:[#6a:2]", 0.0],
        ["[#6a:1]:[#6a:2]", 0.0],
        ["[#8X1,#8X2:1]-[#1:2]", -2.3692047944101415],
        ["[#16:1]-[#8:2]", 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([x[1] * np.sqrt(138.935456) for x in patterns])
    props = None

    am1h = nonbonded.AM1CCCHandler(smirks, params, props)
    obj = am1h.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    assert len(all_handlers) == 1

    new_am1h = all_handlers[0]
    np.testing.assert_equal(new_am1h.smirks, am1h.smirks)
    np.testing.assert_equal(new_am1h.params, am1h.params)
    assert new_am1h.props == am1h.props


def test_lennard_jones_handler():
    patterns = [
        ["[#1:1]", 99.0, 999.0],
        ["[#1:1]-[#6X4]", 99.0, 999.0],
        ["[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X4]~[*+1,*+2]", 99.0, 999.0],
        ["[#1:1]-[#6X3]", 99.0, 999.0],
        ["[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X2]", 99.0, 999.0],
        ["[#1:1]-[#7]", 99.0, 999.0],
        ["[#1:1]-[#8]", 99.0, 999.0],
        ["[#1:1]-[#16]", 99.0, 999.0],
        ["[#6:1]", 0.7, 0.8],
        ["[#6X2:1]", 99.0, 999.0],
        ["[#6X4:1]", 0.1, 0.2],
        ["[#8:1]", 99.0, 999.0],
        ["[#8X2H0+0:1]", 0.5, 0.6],
        ["[#8X2H1+0:1]", 99.0, 999.0],
        ["[#7:1]", 0.3, 0.4],
        ["[#16:1]", 99.0, 999.0],
        ["[#15:1]", 99.0, 999.0],
        ["[#9:1]", 1.0, 1.1],
        ["[#17:1]", 99.0, 999.0],
        ["[#35:1]", 99.0, 999.0],
        ["[#53:1]", 99.0, 999.0],
        ["[#3+1:1]", 99.0, 999.0],
        ["[#11+1:1]", 99.0, 999.0],
        ["[#19+1:1]", 99.0, 999.0],
        ["[#37+1:1]", 99.0, 999.0],
        ["[#55+1:1]", 99.0, 999.0],
        ["[#9X0-1:1]", 99.0, 999.0],
        ["[#17X0-1:1]", 99.0, 999.0],
        ["[#35X0-1:1]", 99.0, 999.0],
        ["[#53X0-1:1]", 99.0, 999.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = None

    ljh = nonbonded.LennardJonesHandler(smirks, params, props)
    obj = ljh.serialize()
    all_handlers, _, _ = deserialize_handlers(bin_to_str(obj))

    ljh = all_handlers[0]
    np.testing.assert_equal(ljh.smirks, ljh.smirks)
    np.testing.assert_equal(ljh.params, ljh.params)
    assert ljh.props == ljh.props
