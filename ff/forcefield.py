import rdkit
from rdkit import Chem

import simtk
import functools

import numpy as np

from openforcefield.utils import toolkits
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import ValenceDict

from timemachine.potentials import bonded


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

def parameterize(mol, forcefield):
    """
    Parameterize an RDKit molecule with a given forcefield.
    """
    # do this in a separate pass later
    global_params = []

    def add_param(p):
        length = len(global_params)
        global_params.append(p)
        return length

    nrg_fns = []

    for handler in forcefield._parameter_handlers.items():
        handler_name, handler_params = handler

        if handler_name == 'Bonds':

            vd = ValenceDict()
            for p in handler_params.parameters:
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, p.smirks)
                for m in matches:
                    vd[m] = (add_param(to_md_units(p.k)), add_param(to_md_units(p.length)))

            bond_idxs = []
            bond_param_idxs = []

            for k, v in vd.items():
                bond_idxs.append(k)
                bond_param_idxs.append(v)

            nrg_fns.append(
                functools.partial(
                    bonded.harmonic_bond,
                    bond_idxs=np.array(bond_idxs),
                    param_idxs=np.array(bond_param_idxs)
                )
            )

        elif handler_name == "Angles":

            vd = ValenceDict()
            for p in handler_params.parameters:
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, p.smirks)
                for m in matches:
                    vd[m] = (add_param(to_md_units(p.k)), add_param(to_md_units(p.angle)))

            angle_idxs = []
            angle_param_idxs = []

            for k, v in vd.items():
                angle_idxs.append(k)
                angle_param_idxs.append(v)

            nrg_fns.append(
                functools.partial(
                    bonded.harmonic_angle,
                    angle_idxs=np.array(angle_idxs),
                    param_idxs=np.array(angle_param_idxs)
                )
            )

        elif handler_name == "ProperTorsions":

            vd = ValenceDict()
            for all_params in handler_params.parameters:
                matches = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, all_params.smirks)
                for m in matches:
                    t_p = []

                    for k, phase, period in zip(all_params.k, all_params.phase, all_params.periodicity):
                        t_p.append((
                            add_param(to_md_units(k)),
                            add_param(to_md_units(phase)),
                            add_param(period),
                        ))
                    vd[m] = t_p

            torsion_idxs = []
            torsion_param_idxs = []

            for k, vv in vd.items():
                for v in vv:
                    torsion_idxs.append(k)
                    torsion_param_idxs.append(v)

            nrg_fns.append(
                functools.partial(
                    bonded.periodic_torsion,
                    torsion_idxs=np.array(torsion_idxs),
                    param_idxs=np.array(torsion_param_idxs)
                )
            )

    return nrg_fns, np.array(global_params)

    print("B", bond_idxs, bond_param_idxs)
    print("A", angle_idxs, angle_param_idxs)
    print("T", torsion_idxs, torsion_param_idxs)
    print(global_params)
    print(nrg_fns)
        # print(handler.name)
        # if force_group == "Bond":
        #     parse_triples(force_group)
        #     bi, pi = match_bonds(mol, triples)
        #     harmonic_bond(conf, global_params, None, bond_idxs, param_idxs)
        # elif force_group == "Angles":
        #     parse_triples(force_group)
        #     match_angles(mol, triples)

    # kb, b0 = global_parameters[kb_idx, b0_idx]


# results = toolkits.RDKitToolkitWrapper._find_smarts_matches(mol, pattern)
# print(results)


# parameterize(Chem.MolFromSmiles("CC1CCCCC1"), forcefield)
