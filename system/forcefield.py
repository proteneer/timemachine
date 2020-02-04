import sys
import rdkit
from rdkit import Chem

import simtk
import functools
import numpy as np

from timemachine import constants

from openforcefield.utils import toolkits
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import ValenceDict

# from timemachine.lib import custom_ops
# from system import custom_functionals
from timemachine.lib import ops

def combiner(
    a_nrgs, b_nrgs,
    a_params, b_params,
    a_param_groups, b_param_groups,
    a_masses, b_masses,
    perm=None):
    """
    Combine two systems with two distinct parameter sets into one.
    """

    num_a_atoms = len(a_masses)                     # offset by number of atoms in a
    c_masses = np.concatenate([a_masses, b_masses]) # combined masses

    if perm is None:
        perm = np.arange(len(c_masses))

    iperm = np.argsort(perm).astype(np.int32)

    perm = np.array(perm, dtype=np.int32)

    np.testing.assert_equal(np.sort(perm), np.arange(len(c_masses))) # assert every atom has a unique permutation

    c_params = np.concatenate([a_params, b_params]) # combined parameters
    c_param_groups = np.concatenate([a_param_groups, b_param_groups]) # combine parameter groups

    assert len(a_nrgs) == len(b_nrgs)

    a_nrgs.sort(key=str)
    b_nrgs.sort(key=str)

    a_nrgs = a_nrgs[::-1]
    b_nrgs = b_nrgs[::-1]

    c_nrgs = []

    for a, b in zip(a_nrgs, b_nrgs):
        a_name = a[0]
        a_args = a[1]
        b_name = b[0]
        b_args = b[1]

        assert a_name == b_name
        assert a_args[-1] == b_args[-1] # dimension
        if a_name == ops.HarmonicBond:
            bond_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
            bond_idxs = iperm[bond_idxs]
            bond_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((ops.HarmonicBond, (bond_idxs, bond_param_idxs, a_args[-1])))
        elif a_name == ops.HarmonicAngle:
            angle_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
            angle_idxs = iperm[angle_idxs]
            angle_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            c_nrgs.append((ops.HarmonicAngle, (angle_idxs, angle_param_idxs, a_args[-1])))
        elif a_name == ops.PeriodicTorsion:
            if len(a_args[0]) > 0:
                torsion_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
                torsion_idxs = iperm[torsion_idxs]
                torsion_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                c_nrgs.append((ops.PeriodicTorsion, (torsion_idxs, torsion_param_idxs, a_args[-1])))
            else:
                c_nrgs.append((ops.PeriodicTorsion, (b_args[0] + num_a_atoms, b_args[1] + len(a_params), b_args[-1])))
            pass
        elif a_name == ops.Nonbonded:
            assert a_args[5] == b_args[5] # cutoff
            assert a_args[-1] == b_args[-1] # dimension

            # directly permute the location of the charges
            es_param_idxs = np.concatenate([a_args[0], b_args[0] + len(a_params)], axis=0) # [N,]
            lj_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
            es_param_idxs = es_param_idxs[perm]
            lj_param_idxs = lj_param_idxs[perm]

            exclusion_idxs = np.concatenate([a_args[2], b_args[2] + num_a_atoms], axis=0)
            exclusion_idxs = iperm[exclusion_idxs] # [E,]

            es_exclusion_param_idxs = np.concatenate([a_args[3], b_args[3] + len(a_params)], axis=0)  # [E, 1]
            lj_exclusion_param_idxs = np.concatenate([a_args[4], b_args[4] + len(a_params)], axis=0)  # [E, 1]

            c_nrgs.append((ops.Nonbonded, (
                es_param_idxs,
                lj_param_idxs,
                exclusion_idxs,
                es_exclusion_param_idxs,
                lj_exclusion_param_idxs,
                a_args[5],
                a_args[-1]
                )
            ))

        else:
            raise Exception("Unknown potential", a_name)

    return c_nrgs, c_params, c_param_groups, c_masses


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
        "[#8:1]": -0.3100,
        "[#8X2H0+0:1]": -0.3700,
        "[#8X2H1+0:1]": -0.3104,
        "[#7:1]": -0.200,
        "[#16:1]": -0.2500,
        "[#15:1]": -0.2000,
        "[#9:1]": -0.361,
        "[#17:1]": -0.265,
        "[#35:1]": -0.083,
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

def parameterize(mol, forcefield, am1=False, dimension=3):
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

    nonbonded_exclusion_idxs = []
    nonbonded_exclusion_params = []
    nonbonded_lj_param_idxs = []
    nonbonded_es_param_idxs = []

    for handler in forcefield._parameter_handlers.items():

        handler_name, handler_params = handler
        if handler_name == 'Bonds':

            vd = ValenceDict()
            for p in handler_params.parameters:
                k_idx, l_idx = add_param(to_md_units(p.k), 0), add_param(to_md_units(p.length), 1)
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, p.smirks)               
                for m in matches:
                    vd[m] = (k_idx, l_idx)

            bond_idxs = []
            bond_param_idxs = []

            for k, v in vd.items():
                bond_idxs.append(k)
                bond_param_idxs.append(v)

            nrg_fns.append((
                ops.HarmonicBond,
                (
                    np.array(bond_idxs, dtype=np.int32),
                    np.array(bond_param_idxs, dtype=np.int32),
                    dimension
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
                ops.HarmonicAngle,
                (
                    np.array(angle_idxs, dtype=np.int32),
                    np.array(angle_param_idxs, dtype=np.int32),
                    dimension
                )
            ))

        elif handler_name == "ImproperTorsions":
            # Disabled for now
            continue # skip while we debug
            vd = ValenceDict()

            for all_params in handler_params.parameters:

                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, all_params.smirks)
                all_k_idxs = []
                all_phase_idxs = []
                all_period_idxs = []

                for k, phase, period in zip(all_params.k, all_params.phase, all_params.periodicity):

                    # (ytz): hack
                    impdivf = 3
                    k_idx, phase_idx, period_idx = add_param(to_md_units(k/impdivf), 4), add_param(to_md_units(phase), 5), add_param(period, 6),
                    all_k_idxs.append(k_idx)
                    all_phase_idxs.append(phase_idx)
                    all_period_idxs.append(period_idx)

                for m in matches:
                    t_p = []
                    for k_idx, phase_idx, period_idx in zip(all_k_idxs, all_phase_idxs, all_period_idxs):
                        t_p.append((k_idx, phase_idx, period_idx))

                    # 3-way trefoil permutation
                    others = [m[0], m[2], m[3]]
                    for p in [(others[i], others[j], others[k]) for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]:
                        vd[(m[1], p[0], p[1], p[2])] = t_p

            torsion_idxs = []
            torsion_param_idxs = []

            for k, vv in vd.items():
                for v in vv:
                    torsion_idxs.append(k)
                    torsion_param_idxs.append(v)

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
                ops.PeriodicTorsion,
                (
                    np.array(torsion_idxs, dtype=np.int32),
                    np.array(torsion_param_idxs, dtype=np.int32),
                    dimension
                )
            ))

        elif handler_name == "vdW":
            # lennardjones
            vd = ValenceDict()
            for param in handler_params.parameters:
                sigma = to_md_units(param.sigma)
                epsilon = to_md_units(param.epsilon)
                s_idx, e_idx = add_param(sigma, 8), add_param(epsilon, 9)
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, param.smirks)
                for m in matches:
                    vd[m] = (s_idx, e_idx)

            for k, v in vd.items():
                nonbonded_lj_param_idxs.append(v)

    if am1:
        assert 0

        # print("Running AM1BCC")

        # mb = Chem.MolToMolBlock(mol)

        # ims = oechem.oemolistream()
        # ims.SetFormat(oechem.OEFormat_SDF)
        # ims.openstring(mb)

        # for buf_mol in ims.GetOEMols():
        #     oemol = oechem.OEMol(buf_mol)

        # result = oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCELF10Charges())

        # if result is False:
        #     raise Exception('Unable to assign charges')

        # partial_charges = []
        # for index, atom in enumerate(oemol.GetAtoms()):
        #     partial_charges.append(atom.GetPartialCharge())

        # partial_charges = np.array(partial_charges)

        # charge_param_idxs = []
        # for charge in partial_charges:
        #     # charge = charge*0.2
        #     c_idx = add_param(charge, 17)
        #     charge_param_idxs.append(c_idx)

        # scale_matrix = np.ones(shape=(num_atoms, num_atoms), dtype=custom_functionals.precision) - np.eye(num_atoms)
        # # fully exclude 1-2, 1-3, tbd: 1-4
        # for (src, dst) in bond_idxs:
        #     scale_matrix[src][dst] = 0
        #     scale_matrix[dst][src] = 0

        # for (src, _, dst) in angle_idxs:
        #     scale_matrix[src][dst] = 0
        #     scale_matrix[dst][src] = 0

        # for (src, _, _, dst) in torsion_idxs:
        #     scale_matrix[src][dst] = 0.83333
        #     scale_matrix[dst][src] = 0.83333

        # nrg_fns.append((
        #     custom_functionals.electrostatics,
        #     (
        #         np.array(scale_matrix, dtype=np.int32), # WRONG
        #         np.array(charge_param_idxs, dtype=np.int32),
        #         custom_functionals.es_cutoff,
        #     )
        # ))

        # print("AM1 partial charges", partial_charges)
        # # assert 0

        # # if ((charges / unit.elementary_charge) == 0.).all():
        # #     # TODO: These will be 0 if the charging failed. What behavior do we want in that case?
        # #     raise Exception(
        # #         "Partial charge calculation failed. Charges from compute_partial_charges() are all 0."
        # #     )
        # #     return charges

    else:

        # process charges separately
        model = simple_charge_model()
        vd = ValenceDict()

        # add parameterize
        for smirks, param in model.items():

            # small charges
            param = param*np.sqrt(constants.ONE_4PI_EPS0)/4

            c_idx = add_param(param, 17)
            matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, smirks)

            for m in matches:
                vd[m] = c_idx

        # charge_param_idxs = []
        for k, v in vd.items():
            nonbonded_es_param_idxs.append(v)

        # print("LIGAND NET CHARGE", np.sum(np.array(global_params)[charge_param_idxs]))
        # guest_charges = np.array(global_params)[charge_param_idxs]
        # offsets = np.sum(guest_charges)/guest_charges.shape[0]

        # for p_idx in set(charge_param_idxs):
        #    global_params[p_idx] -= offsets

        # guest_charges = np.array(global_params)[charge_param_idxs]
        # print("LIGAND NET CHARGE AFTER", guest_charges, "SUM", np.sum(guest_charges))

    exclusion_param_idx = add_param(1.0, 10)

    # insert into a dictionary to avoid double counting exclusions
    exclusions = {}
    for (src, dst) in bond_idxs:
        assert src < dst
        exclusions[(src, dst)] = exclusion_param_idx
    for (src, _, dst) in angle_idxs:
        assert src < dst
        exclusions[(src, dst)] = exclusion_param_idx
    for (src, _, _, dst) in torsion_idxs:
        assert src < dst
        exclusions[(src, dst)] = exclusion_param_idx

    exclusion_idxs = []
    exclusion_param_idxs = []

    for k, v in exclusions.items():
        exclusion_idxs.append(k)
        exclusion_param_idxs.append(v)

    exclusion_idxs = np.array(exclusion_idxs)

    nrg_fns.append((
        ops.Nonbonded,
        (
            np.array(nonbonded_es_param_idxs, dtype=np.int32),
            np.array(nonbonded_lj_param_idxs, dtype=np.int32),
            np.array(exclusion_idxs, dtype=np.int32),
            np.array(exclusion_param_idxs, dtype=np.int32),
            np.array(exclusion_param_idxs, dtype=np.int32),
            0.85,
            dimension
        )
    ))


    # c = mol.GetConformer(0)
    # conf = np.array(c.GetPositions(), dtype=ops.precision)
    # conf = conf/10 # convert to md_units

    masses = []
    for atom in mol.GetAtoms():
        masses.append(atom.GetMass())
    masses = np.array(masses, dtype=np.float64)

    return nrg_fns, (np.array(global_params), np.array(global_param_groups, dtype=np.int32)), masses

