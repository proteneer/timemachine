import sys
import rdkit
from rdkit import Chem

import simtk
import functools

import numpy as np

from openforcefield.utils import toolkits
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import ValenceDict

from timemachine.lib import custom_ops

def merge_potentials(nrgs):
    c_nrgs = []
    for a in nrgs:
        a_name = a[0]
        a_args = a[1]
        if a_name == custom_ops.HarmonicBond_f32:
            c_nrgs.append(custom_ops.HarmonicBond_f32(a_args[0], a_args[1]))
        elif a_name == custom_ops.HarmonicAngle_f32:
            c_nrgs.append(custom_ops.HarmonicAngle_f32(a_args[0], a_args[1]))
        elif a_name == custom_ops.PeriodicTorsion_f32:
            c_nrgs.append(custom_ops.PeriodicTorsion_f32(a_args[0], a_args[1]))
        elif a_name == custom_ops.LennardJones_f32:
            c_nrgs.append(custom_ops.LennardJones_f32(a_args[0].astype(np.float32), a_args[1]))
        elif a_name == custom_ops.Electrostatics_f32:
            c_nrgs.append(custom_ops.Electrostatics_f32(a_args[0].astype(np.float32), a_args[1]))
        else:
            raise Exception("Unknown potential", a_name)

    return c_nrgs  

# todo generalize to N nrg_functionals
def combiner(
    a_nrgs, b_nrgs,
    a_params, b_params,
    a_param_groups, b_param_groups,
    a_conf, b_conf,
    a_masses, b_masses):
    """
    Combine two systems with two distinct parameter sets into one.
    """

    num_a_atoms = len(a_masses)                     # offset by number of atoms in a
    c_masses = np.concatenate([a_masses, b_masses]) # combined masses
    c_conf = np.concatenate([a_conf, b_conf])       # combined geometry
    c_params = np.concatenate([a_params, b_params]) # combined parameters
    c_param_groups = np.concatenate([a_param_groups, b_param_groups]) # combine parameter groups

    assert len(a_nrgs) == len(b_nrgs)

    a_nrgs.sort(key=str)
    b_nrgs.sort(key=str)



    c_nrgs = []
    for a, b in zip(a_nrgs, b_nrgs):
        a_name = a[0]
        a_args = a[1]
        b_name = b[0]
        b_args = b[1]

        assert a_name == b_name
        if a_name == custom_ops.HarmonicBond_f32:
            bond_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
            bond_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((custom_ops.HarmonicBond_f32, (bond_idxs, bond_param_idxs)))
        elif a_name == custom_ops.HarmonicAngle_f32:
            angle_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
            angle_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((custom_ops.HarmonicAngle_f32, (angle_idxs, angle_param_idxs)))
        elif a_name == custom_ops.PeriodicTorsion_f32:
            torsion_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
            torsion_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((custom_ops.PeriodicTorsion_f32, (torsion_idxs, torsion_param_idxs)))
        elif a_name == custom_ops.LennardJones_f32:
            lj_scale_matrix = np.ones(shape=(len(c_masses), len(c_masses)), dtype=np.float64)
            lj_scale_matrix[:num_a_atoms, :num_a_atoms] = a_args[0]
            lj_scale_matrix[num_a_atoms:, num_a_atoms:] = b_args[0]
            lj_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((custom_ops.LennardJones_f32, (lj_scale_matrix, lj_param_idxs)))
        elif a_name == custom_ops.Electrostatics_f32:
            es_scale_matrix = np.ones(shape=(len(c_masses), len(c_masses)), dtype=np.float64)
            es_scale_matrix[:num_a_atoms, :num_a_atoms] = a_args[0]
            es_scale_matrix[num_a_atoms:, num_a_atoms:] = b_args[0]
            es_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((custom_ops.Electrostatics_f32, (es_scale_matrix, es_param_idxs)))
        else:
            raise Exception("Unknown potential", a_name)

    return c_nrgs, c_params, c_param_groups, c_conf, c_masses


def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)


def match_bonds(mol, triples):
    """

    """
    bond_idxs = []
    param_idxs = []

    for smirks, k_idx, length_idx in triples:
        bond_idxs.append([bond.src, bond.dst])
        param_idxs.append([k_idx, length_idx])

    return bond_idxs, param_idxs

def simple_charge_model():
    model = {
        "[#1:1]": 0.0157,
        "[#1:1]-[#6X4]": 0.0157,
        "[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]": 0.0157,
        "[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]": 0.0157,
        "[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]": 0.0157,
        "[#1:1]-[#6X4]~[*+1,*+2]": 0.0157,
        "[#1:1]-[#6X3]": 0.0150,
        "[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]": 0.0150,
        "[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]": 0.0150,
        "[#1:1]-[#6X2]": 0.0150,
        "[#1:1]-[#7]": -0.157,
        "[#1:1]-[#8]": -0.2,
        "[#1:1]-[#16]": 0.0157,
        "[#6:1]": 0.3860,
        "[#6X2:1]": 0.3100,
        "[#6X4:1]": 0.3094,
        "[#8:1]": -0.2100,
        "[#8X2H0+0:1]": -0.1700,
        "[#8X2H1+0:1]": -0.2104,
        "[#7:1]": -0.200,
        "[#16:1]": -0.2500,
        "[#15:1]": -0.2000,
        "[#9:1]": -0.361,
        "[#17:1]": -0.265,
        "[#35:1]": -0.320,
        "[#53:1]": 0.40,
        "[#3+1:1]": 0.0279896,
        "[#11+1:1]": 0.0874393,
        "[#19+1:1]": 0.1936829,
        "[#37+1:1]": 0.3278219,
        "[#55+1:1]": 0.4065394,
        "[#9X0-1:1]": 0.0033640,
        "[#17X0-1:1]": 0.0355910,
        "[#35X0-1:1]": 0.0586554,
        "[#53X0-1:1]": 0.0536816
    }
    return model

def parameterize(mol, forcefield):
    """
    Parameterize an RDKit molecule with a given forcefield.
    """
    # do this in a separate pass later
    global_params = []
    global_param_groups = []

    num_atoms = mol.GetNumAtoms()

    def add_param(p, p_group):
        length = len(global_params)
        global_params.append(p)
        global_param_groups.append(p_group)
        return length

    nrg_fns = []

    for handler in forcefield._parameter_handlers.items():

        handler_name, handler_params = handler
        print(handler_name)

        if handler_name == 'Bonds':

            vd = ValenceDict()
            for p in handler_params.parameters:
                k_idx, l_idx = add_param(to_md_units(p.k), 0), add_param(to_md_units(p.length), 1)
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, p.smirks)
                # print(p.smirks, matches)
                
                for m in matches:
                    vd[m] = (k_idx, l_idx)

            bond_idxs = []
            bond_param_idxs = []

            for k, v in vd.items():
                bond_idxs.append(k)
                bond_param_idxs.append(v)

            nrg_fns.append((
                custom_ops.HarmonicBond_f32,
                (
                    np.array(bond_idxs, dtype=np.int32),
                    np.array(bond_param_idxs, dtype=np.int32)
                )
            ))

        elif handler_name == "Angles":

            vd = ValenceDict()
            for p in handler_params.parameters:
                k_idx, a_idx = add_param(to_md_units(p.k), 2), add_param(to_md_units(p.angle), 3)
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, p.smirks)
                for m in matches:
                    vd[m] = (k_idx, a_idx)

            angle_idxs = []
            angle_param_idxs = []

            for k, v in vd.items():
                angle_idxs.append(k)
                angle_param_idxs.append(v)

            nrg_fns.append((
                custom_ops.HarmonicAngle_f32,
                (
                    np.array(angle_idxs, dtype=np.int32),
                    np.array(angle_param_idxs, dtype=np.int32)
                )
            ))

        # TODO: ImproperTorsions
        elif handler_name == "ProperTorsions":

            vd = ValenceDict()
            for all_params in handler_params.parameters:
                
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, all_params.smirks)

                all_k_idxs = []
                all_phase_idxs = []
                all_period_idxs = []

                for k, phase, period in zip(all_params.k, all_params.phase, all_params.periodicity):
                    k_idx, phase_idx, period_idx = add_param(to_md_units(k), 4), add_param(to_md_units(phase), 5), add_param(period, 6),
                    all_k_idxs.append(k_idx)
                    all_phase_idxs.append(phase_idx)
                    all_period_idxs.append(period_idx)

                for m in matches:
                    t_p = []
                    for k_idx, phase_idx, period_idx in zip(all_k_idxs, all_phase_idxs, all_period_idxs):
                        t_p.append((k_idx, phase_idx, period_idx))
                    vd[m] = t_p

            torsion_idxs = []
            torsion_param_idxs = []

            for k, vv in vd.items():
                for v in vv:
                    torsion_idxs.append(k)
                    torsion_param_idxs.append(v)

            nrg_fns.append((
                custom_ops.PeriodicTorsion_f32,
                (
                    np.array(torsion_idxs, dtype=np.int32),
                    np.array(torsion_param_idxs, dtype=np.int32)
                )
            ))
        elif handler_name == "vdW":
            # lennardjones
            vd = ValenceDict()
            for param in handler_params.parameters:
                s_idx, e_idx = add_param(to_md_units(param.sigma), 8), add_param(to_md_units(param.epsilon), 9)
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, param.smirks)
                for m in matches:
                    vd[m] = (s_idx, e_idx)

                scale_matrix = np.ones(shape=(num_atoms, num_atoms), dtype=np.float64) - np.eye(num_atoms)

                # fully exclude 1-2, 1-3, tbd: 1-4
                for (src, dst) in bond_idxs:
                    scale_matrix[src][dst] = 0
                    scale_matrix[dst][src] = 0

                for (src, _, dst) in angle_idxs:
                    scale_matrix[src][dst] = 0
                    scale_matrix[dst][src] = 0

                for (src, _, _, dst) in torsion_idxs:
                    scale_matrix[src][dst] = 0
                    scale_matrix[dst][src] = 0

            lj_param_idxs = []

            for k, v in vd.items():
                lj_param_idxs.append(v)

            nrg_fns.append((
                custom_ops.LennardJones_f32,
                (
                    np.array(scale_matrix, dtype=np.int32),
                    np.array(lj_param_idxs, dtype=np.int32)
                )
            ))

    # process charges separately
    model = simple_charge_model()
    vd = ValenceDict()
    for smirks, param in model.items():

        c_idx = add_param(param, 7)
        matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, smirks)

        for m in matches:
            vd[m] = c_idx

        scale_matrix = np.ones(shape=(num_atoms, num_atoms), dtype=np.float64) - np.eye(num_atoms)
        # fully exclude 1-2, 1-3, tbd: 1-4
        for (src, dst) in bond_idxs:
            scale_matrix[src][dst] = 0
            scale_matrix[dst][src] = 0

        for (src, _, dst) in angle_idxs:
            scale_matrix[src][dst] = 0
            scale_matrix[dst][src] = 0

        for (src, _, _, dst) in torsion_idxs:
            scale_matrix[src][dst] = 0
            scale_matrix[dst][src] = 0

    charge_param_idxs = []
    for k, v in vd.items():
        charge_param_idxs.append(v)

    nrg_fns.append((
        custom_ops.Electrostatics_f32,
        (
            np.array(scale_matrix, dtype=np.int32),
            np.array(charge_param_idxs, dtype=np.int32)
        )
    ))

    c = mol.GetConformer(0)
    conf = np.array(c.GetPositions(), dtype=np.float64)
    conf = conf/10 # convert to md_units

    masses = []
    for atom in mol.GetAtoms():
        masses.append(atom.GetMass())
    masses = np.array(masses, dtype=np.float64)

    return nrg_fns, np.array(global_params), np.array(global_param_groups, dtype=np.int32), conf, masses

