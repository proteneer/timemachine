from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFMCS
from ff import forcefield
import argparse


def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetSymbol())
    return masses


def order(src, dst):
    assert src != dst
    if src > dst:
        return dst, src
    return src, dst


def reorder(mol, core, offset):
    # use 3d predicates to deal with symmetry/redundant matches later
    core_to_old = np.array(mol.GetSubstructMatches(core)[0])
    old_to_new = [None] * mol.GetNumAtoms()

    for core_idx, target_idx in enumerate(core_to_old):
        old_to_new[target_idx] = core_idx

    counter = core.GetNumAtoms() + offset
    for old_idx, elem in enumerate(old_to_new):
        if elem is None:
            old_to_new[old_idx] = counter
            counter += 1
    return old_to_new


def iterate_type(mol_type, mol_map, type):
    if type == 'HarmonicBond':
        core_bonds = {}
        new_bond_idxs = []
        new_bond_params = []
        for (src, dst), param_idxs in zip(*mol_type):
            param_idxs = tuple(param_idxs.tolist())
            new_src, new_dst = mol_map[src], mol_map[dst]
            key = order(new_src, new_dst)
            if new_src < R_C and new_dst < R_C:
                core_bonds[key] = param_idxs
            else:
                new_bond_idxs.append(key)
                new_bond_params.append(param_idxs)
        return core_bonds, new_bond_idxs, new_bond_params

    elif type == 'HarmonicAngle':
        core_angles = {}
        new_angle_idxs = []
        new_angle_params = []
        for (src, mid, dst), param_idxs in zip(*mol_type):
            param_idxs = tuple(param_idxs.tolist())
            new_src, new_mid, new_dst = mol_map[src], mol_map[mid], mol_map[
                dst]
            key = order(new_src, new_dst)
            if new_src < R_C and new_dst < R_C:
                core_angles[(key[0], new_mid, key[1])] = param_idxs
            else:
                new_angle_idxs.append((key[0], new_mid, key[1]))
                new_angle_params.append(param_idxs)
        return core_angles, new_angle_idxs, new_angle_params

    elif type == 'PeriodicTorsion':
        core_torsions = {}
        new_torsion_idxs = []
        new_torsion_params = []
        for (src, lhs, rhs, dst), param_idxs in zip(*mol_type):
            param_idxs = tuple(param_idxs.tolist())
            new_src, new_lhs, new_rhs, new_dst = mol_map[src], mol_map[
                lhs], mol_map[rhs], mol_map[dst]
            key = order(new_src, new_dst)
            if new_src < R_C and new_dst < R_C:
                core_torsions[(key[0], new_lhs, new_rhs, key[1])] = param_idxs
            else:
                new_torsion_idxs.append((key[0], new_lhs, new_rhs, key[1]))
                new_torsion_params.append(param_idxs)
        return core_torsions, new_torsion_idxs, new_torsion_params

    elif type == 'Nonbonded':
        core_nonbonds = {}
        new_es_params = []
        new_lj_params = []
        new_exclusion_idx = []
        new_es_exclusion_params = []
        new_lj_exclusion_params = []
        for atom_idx, component in enumerate(zip(*mol_type[:-1])):
            #param_idxs = tuple(component.tolist())
            if atom_idx < R_C:
                core_nonbonds[atom_idx] = component
            else:
                new_es_params.append(component[0])
                new_lj_params.append(component[1])
                new_exclusion_idx.append(component[2])
                new_es_exclusion_params.append(component[3])
                new_lj_exclusion_params.append(component[4])

        return core_nonbonds, (new_es_params, new_lj_params, new_exclusion_idx,
                               new_es_exclusion_params,
                               new_lj_exclusion_params)

    elif type == "GBSA":
        core_gbsa = {}
        new_es_params = []
        new_gb_radii = []
        new_gb_scale = []
        for atom_idx, component in enumerate(zip(*mol_type[0:3])):
            if atom_idx < R_C:
                core_gbsa[atom_idx] = component
            else:
                new_es_params.append(component[0])
                new_gb_radii.append(component[1])
                new_gb_scale.append(component[2])

        return core_gbsa, (new_es_params, new_gb_radii, new_gb_scale)


def core_check(core_a, core_b):
    """
    Check k,v in core_a against core_b for consistency. Mismatching bond idxs need to be
    alchemically modified and no longer defined as a core atom.
    """
    good_core = dict()
    new_idxs = []
    new_params = []
    for k, v in core_a.items():
        if v != core_b[k]:
            new_idxs.append(k)
            new_params.append(v)
        else:
            good_core[k] = v

    return good_core, new_idxs, new_params


def core_check_nb(core_a, core_b):
    good_core = dict()
    new_es_params = []
    new_lj_params = []
    new_exclusion_idx = []
    new_es_exclusion_params = []
    new_lj_exclusion_params = []
    for k, v in core_a.items():
        bool = []
        for x, y in zip(v, core_b[k]):
            if isinstance(x, int):
                bool.append((x != y))
            else:
                bool.append((x.any() != y.any()))
        if any(bool):
            new_es_params.append(v[0])
            new_lj_params.append(v[1])
            new_exclusion_idx.append(v[2])
            new_es_exclusion_params.append(v[3])
            new_lj_exclusion_params.append(v[4])
        else:
            good_core[k] = v
    return good_core, (new_es_params, new_lj_params, new_exclusion_idx,
                       new_es_exclusion_params, new_lj_exclusion_params)


def core_check_gbsa(core_a, core_b):
    good_core = dict()
    new_es_params = []
    new_gb_radii = []
    new_gb_scale = []
    for k, v in core_a.items():
        if v != core_b[k]:
            new_es_params.append(v[0])
            new_gb_radii.append(v[1])
            new_gb_scale.append(v[2])
        else:
            good_core[k] = v
    return good_core, (new_es_params, new_gb_radii, new_gb_scale)


parser = argparse.ArgumentParser(description='Quick Test')
parser.add_argument('--ligand_sdf', type=str, required=True, nargs="*")
args = parser.parse_args()

off = forcefield.Forcefield("ff/smirnoff_1.1.0.py")

for ligand_sdf_file in args.ligand_sdf:
    suppl = Chem.SDMolSupplier(ligand_sdf_file, removeHs=False)

all_guest_mols = []
for guest_idx, guest_mol in enumerate(suppl):
    all_guest_mols.append(guest_mol)

mol1 = all_guest_mols[0]
mol2 = all_guest_mols[1]
m1_masses = get_masses(mol1)
m2_masses = get_masses(mol2)

core_pattern = rdFMCS.FindMCS([mol1, mol2]).smartsString
core = Chem.MolFromSmarts(core_pattern)

R_C = core.GetNumAtoms()
R_A = mol1.GetNumAtoms() - core.GetNumAtoms()
R_B = mol2.GetNumAtoms() - core.GetNumAtoms()

# bonds, angles, torsions, lj, nb, exclusions, GB, will all need to use this new mapping
mol1_map = reorder(mol1, core, offset=0)
mol2_map = reorder(mol2, core, offset=R_A)

new_masses = [None] * (R_C + R_A + R_B)
for src, dst in enumerate(mol1_map):
    new_masses[dst] = m1_masses[src]
for src, dst in enumerate(mol2_map):
    new_masses[dst] = m2_masses[src]

#print("combined masses", len(new_masses), new_masses)

for type in [
        'HarmonicBond', 'HarmonicAngle', 'PeriodicTorsion', 'Nonbonded', 'GBSA'
]:

    if type == 'Nonbonded':
        lambda_idxs = []

        mol1_type = off.parameterize(mol1)[type]
        mol1_core_type, mol1_component = iterate_type(mol1_type, mol1_map,
                                                      type)

        mol2_type = off.parameterize(mol2)[type]
        mol2_core_type, mol2_component = iterate_type(mol2_type, mol2_map,
                                                      type)

        gc1, c1 = core_check_nb(mol1_core_type, mol2_core_type)
        gc2, c2 = core_check_nb(mol2_core_type, mol1_core_type)

        # ensure the cores are consistent
        #assert gc1.any() == gc2.any()

        for l in range(len(mol1_component)):
            mol1_component[l].extend(c1[l])
            mol2_component[l].extend(c2[l])

        combined_es_params = []
        combined_lj_params = []
        combined_exclusion_idx = []
        combined_es_exclusion_params = []
        combined_lj_exclusion_params = []
        cutoff = mol1_type[-1]
        print('cutoff = ', cutoff)

        for k, v in gc1.items():
            lambda_idxs.append(0)
            combined_es_params.append(v[0])
            combined_lj_params.append(v[1])
            combined_exclusion_idx.append(v[2])
            combined_es_exclusion_params.append(v[3])
            combined_lj_exclusion_params.append(v[4])

        for v in zip(*mol1_component):
            lambda_idxs.append(1)
            combined_es_params.append(v[0])
            combined_lj_params.append(v[1])
            combined_exclusion_idx.append(v[2])
            combined_es_exclusion_params.append(v[3])
            combined_lj_exclusion_params.append(v[4])

        for k, v in gc1.items():
            lambda_idxs.append(0)
            combined_es_params.append(v[0])
            combined_lj_params.append(v[1])
            combined_exclusion_idx.append(v[2])
            combined_es_exclusion_params.append(v[3])
            combined_lj_exclusion_params.append(v[4])

        for v in zip(*mol2_component):
            lambda_idxs.append(-1)
            combined_es_params.append(v[0])
            combined_lj_params.append(v[1])
            combined_exclusion_idx.append(v[2])
            combined_es_exclusion_params.append(v[3])
            combined_lj_exclusion_params.append(v[4])

        # for l_idx, a, b, c, d, e in zip(lambda_idxs, combined_es_params,
        #                                 combined_lj_params,
        #                                 combined_exclusion_idx,
        #                                 combined_es_exclusion_params,
        #                                 combined_lj_exclusion_params):

        #     print(l_idx)
        #     print(a, b, c, d, e)

    elif type == "GBSA":
        lambda_idxs = []

        mol1_type = off.parameterize(mol1)[type]
        mol1_core_type, mol1_component = iterate_type(mol1_type, mol1_map,
                                                      type)

        mol2_type = off.parameterize(mol2)[type]
        mol2_core_type, mol2_component = iterate_type(mol2_type, mol2_map,
                                                      type)

        gc1, c1 = core_check_nb(mol1_core_type, mol2_core_type)
        gc2, c2 = core_check_nb(mol2_core_type, mol1_core_type)

        # ensure the cores are consistent
        #assert gc1.any() == gc2.any()

        for l in range(len(mol1_component)):
            mol1_component[l].extend(c1[l])
            mol2_component[l].extend(c2[l])

        combined_es_params = []
        combined_gb_radii = []
        combined_gb_scale = []
        gb_args = mol1_type[3:]
        print(gb_args)

        for k, v in gc1.items():
            lambda_idxs.append(0)
            combined_es_params.append(v[0])
            combined_gb_radii.append(v[1])
            combined_gb_scale.append(v[2])

        for v in zip(*mol1_component):
            lambda_idxs.append(1)
            combined_es_params.append(v[0])
            combined_gb_radii.append(v[1])
            combined_gb_scale.append(v[2])

        for k, v in gc1.items():
            lambda_idxs.append(0)
            combined_es_params.append(v[0])
            combined_gb_radii.append(v[1])
            combined_gb_scale.append(v[2])

        for v in zip(*mol2_component):
            lambda_idxs.append(-1)
            combined_es_params.append(v[0])
            combined_gb_radii.append(v[1])
            combined_gb_scale.append(v[2])

        for l_idx, a, b, c in zip(lambda_idxs, combined_es_params,
                                  combined_gb_radii, combined_gb_scale):

            print(l_idx)
            print(a, b, c)

    else:
        lambda_idxs = []

        mol1_type = off.parameterize(mol1)[type]
        mol1_core_type, mol1_new_type_idxs, mol1_new_type_params = iterate_type(
            mol1_type, mol1_map, type)

        mol2_type = off.parameterize(mol2)[type]
        mol2_core_type, mol2_new_type_idxs, mol2_new_type_params = iterate_type(
            mol2_type, mol2_map, type)

        gc1, b1, p1 = core_check(mol1_core_type, mol2_core_type)
        gc2, b2, p2 = core_check(mol2_core_type, mol1_core_type)

        # ensure the cores are consistent
        assert gc1 == gc2

        mol1_new_type_idxs.extend(b1)
        mol1_new_type_params.extend(p1)

        mol2_new_type_idxs.extend(b2)
        mol2_new_type_params.extend(p2)

        if type == 'HarmonicBond':
            combined_bond_idxs = []
            combined_bond_params = []

            for k, v in gc1.items():
                lambda_idxs.append(0)
                combined_bond_idxs.append(k)
                combined_bond_params.append(v)

            for k, v in zip(mol1_new_type_idxs, mol1_new_type_params):
                lambda_idxs.append(1)
                combined_bond_idxs.append(k)
                combined_bond_params.append(v)

            for k, v in gc1.items():
                lambda_idxs.append(0)
                combined_bond_idxs.append(k)
                combined_bond_params.append(v)

            for k, v in zip(mol2_new_type_idxs, mol2_new_type_params):
                lambda_idxs.append(-1)
                combined_bond_idxs.append(k)
                combined_bond_params.append(v)

            # for l_idx, bond_idxs, bond_params in zip(lambda_idxs,
            #                                          combined_bond_idxs,
            #                                          combined_bond_params):
            #     print(l_idx, new_masses[bond_idxs[0]], new_masses[bond_idxs[1]],
            #           bond_params)

        elif type == 'HarmonicAngle':
            combined_angle_idxs = []
            combined_angle_params = []

            for k, v in gc1.items():
                lambda_idxs.append(0)
                combined_angle_idxs.append(k)
                combined_angle_params.append(v)

            for k, v in zip(mol1_new_type_idxs, mol1_new_type_params):
                lambda_idxs.append(1)
                combined_angle_idxs.append(k)
                combined_angle_params.append(v)

            for k, v in gc1.items():
                lambda_idxs.append(0)
                combined_angle_idxs.append(k)
                combined_angle_params.append(v)

            for k, v in zip(mol2_new_type_idxs, mol2_new_type_params):
                lambda_idxs.append(-1)
                combined_angle_idxs.append(k)
                combined_angle_params.append(v)

            # for l_idx, angle_idxs, angle_params in zip(lambda_idxs,
            #                                            combined_angle_idxs,
            #                                            combined_angle_params):
            #     print(l_idx, new_masses[angle_idxs[0]], new_masses[angle_idxs[1]],
            #           new_masses[angle_idxs[2]], angle_params)

        elif type == 'PeriodicTorsion':
            combined_torsion_idxs = []
            combined_torsion_params = []

            for k, v in gc1.items():
                lambda_idxs.append(0)
                combined_torsion_idxs.append(k)
                combined_torsion_params.append(v)

            for k, v in zip(mol1_new_type_idxs, mol1_new_type_params):
                lambda_idxs.append(1)
                combined_torsion_idxs.append(k)
                combined_torsion_params.append(v)

            for k, v in gc1.items():
                lambda_idxs.append(0)
                combined_torsion_idxs.append(k)
                combined_torsion_params.append(v)

            for k, v in zip(mol2_new_type_idxs, mol2_new_type_params):
                lambda_idxs.append(-1)
                combined_torsion_idxs.append(k)
                combined_torsion_params.append(v)

            # for l_idx, torsion_idxs, torsion_params in zip(
            #         lambda_idxs, combined_torsion_idxs, combined_torsion_params):
            #     print(l_idx, new_masses[torsion_idxs[0]], torsion_idxs,
            #           new_masses[torsion_idxs[1]], new_masses[torsion_idxs[2]],
            #           new_masses[torsion_idxs[3]], torsion_params)
