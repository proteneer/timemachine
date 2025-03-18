import dataclasses
import os
from glob import glob
from pathlib import Path
from tempfile import NamedTemporaryFile
from warnings import catch_warnings

import numpy as np
import pytest
from common import load_split_forcefields, temporary_working_dir
from openmm import app
from openmm import openmm as mm
from rdkit import Chem

from timemachine import constants
from timemachine.ff import Forcefield, combine_params
from timemachine.ff.handlers import nonbonded
from timemachine.ff.handlers.deserialize import deserialize_handlers
from timemachine.ff.handlers.openmm_deserializer import deserialize_system
from timemachine.md import builders
from timemachine.potentials.potentials import PeriodicTorsion
from timemachine.utils import path_to_internal_file

pytestmark = [pytest.mark.nocuda]


def test_empty_ff():
    # should not throw exception
    ff = Forcefield.from_handlers([])
    ff.serialize()


def test_serialization_of_ffs():
    for path in glob("timemachine/ff/params/smirnoff_*.py"):
        handlers, protein_ff, water_ff = deserialize_handlers(open(path).read())
        ff = Forcefield.from_handlers(handlers, protein_ff=protein_ff, water_ff=water_ff)
        assert ff.protein_ff == constants.DEFAULT_PROTEIN_FF
        assert ff.water_ff == constants.DEFAULT_WATER_FF
        for handler_name, handler in dataclasses.asdict(ff).items():
            if handler_name == "env_bcc_handle":
                assert handler is None
            else:
                assert handler is not None


def test_loading_forcefield_from_file():
    builtin_ffs = glob("timemachine/ff/params/smirnoff_*.py")
    for path in builtin_ffs:
        # Use the full path
        ff = Forcefield.load_from_file(path)
        assert ff is not None
        # Use full path as Path object
        ff = Forcefield.load_from_file(Path(path))
        assert ff is not None
        with catch_warnings(record=True) as w:
            # Load using just file name of the built in
            ff = Forcefield.load_from_file(os.path.basename(path))
            assert ff is not None
        assert len(w) == 0

    for prefix in ["", "timemachine/ff/params/"]:
        path = os.path.join(prefix, "nosuchfile.py")
        with pytest.raises(ValueError) as e:
            Forcefield.load_from_file(path)
        assert path in str(e.value)
        with pytest.raises(ValueError):
            Forcefield.load_from_file(Path(path))
        assert path in str(e.value)

    with temporary_working_dir():
        # Verify that if a local file shadows a builtin
        for path in builtin_ffs:
            basename = os.path.basename(path)
            with open(basename, "w") as ofs:
                ofs.write("junk")
            with catch_warnings(record=True) as w:
                Forcefield.load_from_file(basename)
            assert len(w) == 1
            assert basename in str(w[0].message)


def test_load_default():
    """assert that load_default() is an alias for load_from_file(DEFAULT_FF)"""

    ref = Forcefield.load_from_file(constants.DEFAULT_FF)
    test = Forcefield.load_default()

    # ref == test evaluates to false, so assert equality of smirks lists and parameter arrays manually

    ref_handles = ref.get_ordered_handles()
    test_handles = test.get_ordered_handles()

    assert len(ref_handles) == len(test_handles)

    for ref_handle, test_handle in zip(ref.get_ordered_handles(), test.get_ordered_handles()):
        if ref_handle is None:
            assert test_handle is None
            continue
        assert ref_handle.smirks == test_handle.smirks
        np.testing.assert_array_equal(ref_handle.params, test_handle.params)


def test_serialize_load_precomputed_default():
    """Assert that we can serialize/deserialize the precomputed default forcefield"""
    ff = Forcefield.load_precomputed_default()
    with NamedTemporaryFile(suffix=".py") as temp:
        with open(temp.name, "w") as ofs:
            ofs.write(ff.serialize())
        deserialized_ff = Forcefield.load_from_file(temp.name)
    assert isinstance(ff.q_handle, nonbonded.PrecomputedChargeHandler)
    assert isinstance(deserialized_ff.q_handle, nonbonded.PrecomputedChargeHandler)


def test_split():
    ffs = load_split_forcefields()

    def check(ff):
        params = ff.get_params()
        np.testing.assert_array_equal(ff.hb_handle.params, params.hb_params)
        np.testing.assert_array_equal(ff.ha_handle.params, params.ha_params)
        np.testing.assert_array_equal(ff.pt_handle.params, params.pt_params)
        np.testing.assert_array_equal(ff.it_handle.params, params.it_params)
        np.testing.assert_array_equal(ff.q_handle.params, params.q_params)
        np.testing.assert_array_equal(ff.q_handle_intra.params, params.q_params_intra)
        np.testing.assert_array_equal(ff.lj_handle.params, params.lj_params)
        np.testing.assert_array_equal(ff.lj_handle_intra.params, params.lj_params_intra)

        assert ff.get_ordered_handles() == [
            ff.hb_handle,
            ff.ha_handle,
            ff.pt_handle,
            ff.it_handle,
            ff.q_handle,
            ff.q_handle_intra,
            ff.lj_handle,
            ff.lj_handle_intra,
            ff.env_bcc_handle,  # None
        ]

        combined = combine_params(ff.get_params(), ff.get_params())
        np.testing.assert_array_equal((ff.hb_handle.params, ff.hb_handle.params), combined.hb_params)
        np.testing.assert_array_equal((ff.ha_handle.params, ff.ha_handle.params), combined.ha_params)
        np.testing.assert_array_equal((ff.pt_handle.params, ff.pt_handle.params), combined.pt_params)
        np.testing.assert_array_equal((ff.it_handle.params, ff.it_handle.params), combined.it_params)
        np.testing.assert_array_equal((ff.q_handle.params, ff.q_handle.params), combined.q_params)
        np.testing.assert_array_equal((ff.q_handle_intra.params, ff.q_handle_intra.params), combined.q_params_intra)
        np.testing.assert_array_equal((ff.lj_handle.params, ff.lj_handle.params), combined.lj_params)
        np.testing.assert_array_equal((ff.lj_handle_intra.params, ff.lj_handle_intra.params), combined.lj_params_intra)

    check(ffs.ref)
    check(ffs.intra)
    check(ffs.env)
    check(ffs.scaled)


def test_amber14_tip3p_matches_tip3p():
    """Verify that given a water box, the same parameters are produced for amber14/tip3p as tip3p, but with additional
    support for Ions"""
    tip3p_water_ff = "tip3p"
    assert constants.DEFAULT_WATER_FF != tip3p_water_ff
    ref_host_config = builders.build_water_system(4.0, constants.DEFAULT_WATER_FF)
    tip3p_host_config = builders.build_water_system(4.0, tip3p_water_ff)
    np.testing.assert_array_equal(ref_host_config.masses, tip3p_host_config.masses)
    ref_pots = ref_host_config.host_system.get_U_fns()
    test_pots = tip3p_host_config.host_system.get_U_fns()
    assert len(ref_pots) == len(test_pots)
    for ref, test in zip(ref_pots, test_pots):
        np.testing.assert_array_equal(ref.params, test.params)

    mol = Chem.MolFromSmiles("[Zn+2]")
    with NamedTemporaryFile(suffix=".pdb") as temp:
        Chem.MolToPDBFile(mol, temp.name)
        # tip3p will fail to handle ions
        with pytest.raises(ValueError, match="No template found for residue 1"):
            builders.build_protein_system(temp.name, constants.DEFAULT_PROTEIN_FF, tip3p_water_ff)

        # Amber14/tip3p handles ions without issue
        builders.build_protein_system(temp.name, constants.DEFAULT_PROTEIN_FF, constants.DEFAULT_WATER_FF)


def test_openmm_deserialize_system_handles_duplicate_bonded_forces():
    """In some cases it may be useful to construct an OpenMM system that contains forces of the same type.
    Expectation is that these forces get correctly converted to Timemachine bound potentials. Test only exercises
    Periodic Torsions, but expected to work for bonds/angles.

    Note that we do not handle duplicate Nonbonded potentials, mostly to avoid nuances of merging them.
    """
    ff = app.ForceField(f"{constants.DEFAULT_PROTEIN_FF}.xml")

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_pdb = app.PDBFile(str(path_to_pdb))

    # Create empty topology and coordinates.
    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)

    omm_host_system = ff.createSystem(
        modeller.getTopology(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )

    assert len([f for f in omm_host_system.getForces() if isinstance(f, mm.PeriodicTorsionForce)]) == 1

    existing_torsion_force = next(f for f in omm_host_system.getForces() if isinstance(f, mm.PeriodicTorsionForce))
    first_torsion = existing_torsion_force.getTorsionParameters(0)

    initial_num_torsions = existing_torsion_force.getNumTorsions()
    total_torsions = initial_num_torsions

    torsion_idxs = first_torsion[:4]
    period = 3
    phase = 180.0
    k = 10.0

    # Add a new force that duplicates an existing torsion, the parameters of the torsion is unimportant
    new_force = mm.PeriodicTorsionForce()
    new_force.setName("dihedrals_a")
    new_force.addTorsion(*torsion_idxs, period, phase, k)
    omm_host_system.addForce(new_force)
    total_torsions += new_force.getNumTorsions()

    assert len([f for f in omm_host_system.getForces() if isinstance(f, mm.PeriodicTorsionForce)]) == 2

    bps, _ = deserialize_system(omm_host_system, cutoff=1.2)
    # We separate the torsions in the OpenMM system into periodic/improper, have to get both and combine indices
    tm_torsions = [bp.potential for bp in bps if isinstance(bp.potential, PeriodicTorsion)]
    tm_torsion_idxs = np.concatenate([pot.idxs for pot in tm_torsions])
    assert tm_torsion_idxs.shape == (total_torsions, 4)
