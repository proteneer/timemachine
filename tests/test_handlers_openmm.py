# test that vjps in the OpenMM ff handler is behaving correctly
import functools

import jax
import numpy as np

from simtk import unit
from simtk.openmm import app, Vec3
from ff.handlers import debug_omm


def strip_units(coords):
    return unit.Quantity(np.array(coords / coords.unit), coords.unit)


def test_openmm_handler():

    ref_protein_ff = app.ForceField('ff/params/amber99sbildn.xml',  'ff/params/tip3p.xml')
    protein_ff = debug_omm.ForceField('ff/params/amber99sbildn.xml', 'ff/params/tip3p.xml')

    host_pdb = app.PDBFile('tests/data/hif2a_nowater_min.pdb')

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)
    padding = 1.0
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)
    box_lengths = box_lengths.value_in_unit_system(unit.md_unit_system)
    box_lengths = box_lengths+padding
    box = np.eye(3, dtype=np.float64)*box_lengths
    modeller.addSolvent(ref_protein_ff, boxSize=np.diag(box)*unit.nanometers, neutralize=False)

    hb_handle = None
    ha_handle = None
    pt_handle = None
    nb_handle = None

    for f in protein_ff._forces:
        if isinstance(f, debug_omm.HarmonicBondGenerator):
            assert hb_handle is None
            hb_handle = f
        elif isinstance(f, debug_omm.HarmonicAngleGenerator):
            assert ha_handle is None
            ha_handle = f
        elif isinstance(f, debug_omm.PeriodicTorsionGenerator):
            assert pt_handle is None
            pt_handle = f
        elif isinstance(f, debug_omm.NonbondedGenerator):
            assert nb_handle is None
            nb_handle = f
        else:
            print("f", f)
            raise ValueError("Unknown Handler")

    assert hb_handle is not None
    assert ha_handle is not None
    assert pt_handle is not None
    assert nb_handle is not None

    sys_data = protein_ff.createSystemData(
        modeller.topology
    )

    hb_handle.params = np.array(hb_handle.params)
    ha_handle.params = np.array(ha_handle.params)
    nb_handle.typed_params.params = np.array(nb_handle.typed_params.params)

    hb_handle.parameterize(hb_handle.params, sys_data)

    bond_params, bond_vjp_fn, bond_idxs = jax.vjp(functools.partial(hb_handle.parameterize, data=sys_data), hb_handle.params, has_aux=True)

    ha_handle.parameterize(ha_handle.params, sys_data)

    angle_params, angle_vjp_fn, angle_idxs = jax.vjp(functools.partial(ha_handle.parameterize, data=sys_data), ha_handle.params, has_aux=True)

    pt_handle.parameterize(pt_handle.params, sys_data)

    torsion_params, torsion_vjp_fn, torsion_idxs = jax.vjp(functools.partial(pt_handle.parameterize, data=sys_data), pt_handle.params, has_aux=True)

    nb_handle.parameterize(nb_handle.typed_params.params, sys_data)

    nb_params, nb_vjp_fn, exc_info = jax.vjp(functools.partial(nb_handle.parameterize, data=sys_data), pt_handle.params, has_aux=True)

    # print(exc_info)

    # nb_handle.generateExclusions()