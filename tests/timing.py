import functools

import numpy as np

from simtk.openmm import app
from simtk import openmm as mm
from simtk import unit

from timemachine.potentials import bonded
from timemachine.kernels import custom_ops

from jax.config import config; config.update("jax_enable_x64", True)
import jax


def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)


def create_system(file_path):
    ff = app.ForceField('amber99sb.xml', 'amber99_obc.xml')
    pdb = app.PDBFile(file_path)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )

    global_params = []

    def upsert_parameter(obj):
        """
        Attempts to insert a value v and return the index it belongs to
        """
        if obj not in global_params:
            global_params.append(obj)
            return len(global_params) - 1
        else:
            for v_idx, v in enumerate(global_params):
                if v == obj:
                    return v_idx

    ref_potentials = []
    test_potentials = []

    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):

            bond_idxs = []
            param_idxs = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)
                k = value(k)

                k_idx = upsert_parameter(k)
                b_idx = upsert_parameter(length)

                param_idxs.append([k_idx, b_idx])
                bond_idxs.append([src_idx, dst_idx])

            bond_idxs = np.array(bond_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            ref_hb = functools.partial(
                bonded.harmonic_bond,
                bond_idxs=bond_idxs,
                param_idxs=param_idxs,
                box=None
            )

            test_hb = custom_ops.HarmonicBond_f64(
                bond_idxs,
                param_idxs
            )

            ref_potentials.append(ref_hb)
            test_potentials.append(test_hb)

        if isinstance(force, mm.HarmonicAngleForce):

            angle_idxs = []
            param_idxs = []

            for a_idx in range(force.getNumAngles()):
                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)

                k_idx = upsert_parameter(k)
                a_idx = upsert_parameter(angle)

                param_idxs.append([k_idx, a_idx])
                # print(src_idx, mid_idx, dst_idx)
                angle_idxs.append([src_idx, mid_idx, dst_idx])

            angle_idxs = np.array(angle_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            ref_ha = functools.partial(
                bonded.harmonic_angle,
                angle_idxs=angle_idxs,
                param_idxs=param_idxs,
                box=None
            )

            test_ha = custom_ops.HarmonicAngle_f64(
                angle_idxs,
                param_idxs
            )

            ref_potentials.append(ref_ha)
            test_potentials.append(test_ha)

        if isinstance(force, mm.PeriodicTorsionForce):

            torsion_idxs = []
            param_idxs = []

            for t_idx in range(force.getNumTorsions()):
                a_idx, b_idx, c_idx, d_idx, period, phase, k = force.getTorsionParameters(t_idx)

                # period is unitless
                phase = value(phase)
                k = value(k)

                k_idx = upsert_parameter(k)
                phase_idx = upsert_parameter(phase)
                period_idx = upsert_parameter(period)

                param_idxs.append([k_idx, phase_idx, period_idx])
                torsion_idxs.append([a_idx, b_idx, c_idx, d_idx])

            torsion_idxs = np.array(torsion_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            ref_ha = functools.partial(
                bonded.periodic_torsion,
                torsion_idxs=torsion_idxs,
                param_idxs=param_idxs,
                box=None
            )

            test_ha = custom_ops.PeriodicTorsion_f64(
                torsion_idxs,
                param_idxs
            )

            ref_potentials.append(ref_ha)
            test_potentials.append(test_ha)


    return ref_potentials, test_potentials, np.array(value(pdb.positions), dtype=np.float64), np.array(global_params, np.float64)

all_ref, all_test, coords, params = create_system("/home/yutong/Code/openmm/examples/5dfr_minimized.pdb")

def batch_mult_jvp(fn, x, p, dxdp):
    dpdp = np.eye(p.shape[0])
    def apply_one(dxdp_i, dpdp_i):
        return jax.jvp(
            fn,
            (x, p),
            (dxdp_i, dpdp_i)
        )
    a, b = jax.vmap(apply_one)(dxdp, dpdp)
    return a[0], b

def test_energy(ref_e_fn, test_e_fn, coords, params):

    print("comparing", ref_e_fn, "against", test_e_fn)

    dxdp = np.random.rand(params.shape[0], coords.shape[0], coords.shape[1])
    ref_e = ref_e_fn(coords, params)
    ref_de_dx_fn = jax.grad(ref_e_fn, argnums=(0,))
    ref_de_dx = ref_de_dx_fn(coords, params)
    _, ref_de_dp_jvp = batch_mult_jvp(ref_e_fn, coords, params, dxdp)
    _, ref_d2e_dxdp_jvp = batch_mult_jvp(ref_de_dx_fn, coords, params, dxdp)

    dxdp = np.expand_dims(dxdp, axis=0)
    coords = np.expand_dims(coords, axis=0)

    test_e, test_de_dx, test_de_dp_jvp, test_d2e_dxdp_jvp = test_e_fn.derivatives(coords, params, dxdp, np.arange(len(params), dtype=np.int32))

    np.testing.assert_almost_equal(ref_e, test_e)
    np.testing.assert_almost_equal(ref_de_dx, test_de_dx)
    np.testing.assert_almost_equal(ref_de_dp_jvp, test_de_dp_jvp[0])
    # test_energies(all_ref, all_test, coords, params)

for ref_e_fn, test_e_fn in zip(all_ref, all_test):
    test_energy(ref_e_fn, test_e_fn, coords, params)

