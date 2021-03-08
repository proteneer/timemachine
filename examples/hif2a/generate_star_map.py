# Construct a star map for the fep-benchmark hif2a ligands
import sys
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
from functools import partial

from pickle import dump

import timemachine
from timemachine.parser import TimemachineConfig

from fe import topology
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe.free_energy import RelativeFreeEnergy
from fe.topology import AtomMappingError

# 0. Get force field
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from rdkit import Chem
from rdkit.Chem import rdFMCS

import numpy as np
from fe.atom_mapping import (
    get_core_by_geometry,
    get_core_by_matching,
    get_core_by_mcs,
    get_core_by_smarts,
    mcs_map,
    transformation_size,
    get_star_map,
)

# Get the root off of the timemachine install
root = Path(timemachine.__file__).absolute().parent


def get_mol_id(mol, mol_prop):
    return mol.GetProp(mol_prop)


def _compute_label(mol_a, mol_b, prop_name: str):
    """ Compute labeled ddg (in kJ/mol) from the experimental IC50 s """

    try:
        label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp(prop_name)))
        label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp(prop_name)))
    except KeyError as e:
        raise RuntimeError(f"Couldn't access IC50 label for either mol A or mol B, looking at {prop_name}")

    label = label_dG_b - label_dG_a

    return label


def rbfe_transformation_estimate(rfe: RelativeFreeEnergy):
    return transformation_size(rfe.mol_a.GetNumAtoms(), rfe.mol_b.GetNumAtoms(), len(rfe.core))


def get_core_by_permissive_mcs(mol_a, mol_b):
    mcs_result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=30,
        threshold=1.0,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        completeRingsOnly=True,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=True,
    )
    query = mcs_result.queryMol

    # fails distance assertions
    # return get_core_by_mcs(mol_a, mol_b, query)

    inds_a = mol_a.GetSubstructMatches(query)[0]
    inds_b = mol_b.GetSubstructMatches(query)[0]
    core = np.array([inds_a, inds_b]).T

    return core


def _get_match(mol, query):
    matches = mol.GetSubstructMatches(query)
    return matches[0]


def _get_core_by_smarts_wo_checking_uniqueness(mol_a, mol_b, core_smarts):
    """no atom mapping errors with this one, but the core size is smaller"""
    query = Chem.MolFromSmarts(core_smarts)

    return np.array([_get_match(mol_a, query), _get_match(mol_b, query)]).T

core_strategies = {
    'custom_mcs': lambda a, b : get_core_by_mcs(a, b, mcs_map(a, b).queryMol),
    'any_mcs': lambda a, b : get_core_by_permissive_mcs(a, b),
    'geometry': lambda a, b: get_core_by_geometry(a, b, threshold=0.5),
    'smarts_core_1': lambda a, b: get_core_by_smarts(a, b, core_smarts=bicyclic_smarts_pattern),
    'smarts_core_2': lambda a, b: _get_core_by_smarts_wo_checking_uniqueness(a, b, core_smarts=core_2_smarts)
}


def generate_star(
        hub,
        spokes,
        forcefield,
        transformation_size_threshold: int = 2,
        core_strategy: str = 'geometry',
        label_property: str="IC50[uM](SPA)"):
    transformations = []
    error_transformations = []
    for spoke in spokes:
        core = core_strategies[core_strategy](hub, spoke)

        # TODO: reduce overlap between get_core_by_smarts and get_core_by_mcs
        # TODO: replace big ol' list of get_core_by_*(mol_a, mol_b, **kwargs) functions with something... classy

        try:
            single_topology = topology.SingleTopology(hub, spoke, core, forcefield)
            rfe = RelativeFreeEnergy(single_topology, label=_compute_label(hub, spoke, label_property))
            transformations.append(rfe)
        except AtomMappingError as e:
            # note: some of transformations may fail the factorizability assertion here:
            # https://github.com/proteneer/timemachine/blob/2eb956f9f8ce62287cc531188d1d1481832c5e96/fe/topology.py#L381-L431
            error_transformations.append((hub, spoke, core))

    print(f'total # of edges that encountered atom mapping errors: {len(error_transformations)}')

    # filter to keep just the edges with very small number of atoms changing
    easy_transformations = [rfe for rfe in transformations if rbfe_transformation_estimate(rfe) <= transformation_size_threshold]

    return easy_transformations, error_transformations

def mol_matches_core(mol, core_query) -> bool:
    res = mol.GetSubstructMatches(core_query)
    if len(res) > 1:
        print(f"Mol matched core multiple times")
    return len(res) == 1

def smarts_comparison(smarts: str):
    core_query = Chem.MolFromSmarts(smarts)

    def matching_mols(mols):
        matches = []
        leftover = []
        for mol in mols:
            res = mol.GetSubstructMatches(core_query)
            if len(res) > 1:
                print(f"Mol matched core multiple times")
            if len(res) == 1:
                matches.append(mol)
            else:
                leftover.append(mol)
        return matches, leftover
    return matching_mols


core_generation_methods = {
    "smarts": smarts_comparison,
}

def manual_hub_selection(property: str, value: str):
    def find_hub(mols):
        hub = None
        idx = -1
        for i, mol in enumerate(mols):
            if mol.GetProp(property) == value:
                idx = i
                hub = mol
                break
        if hub is not None:
            mols.pop(idx)
        return hub, mols
    return find_hub


def mcs_star_map(threshold: float = 0.5):
    return partial(get_star_map, threshold=threshold)


network_generation_methods = {
    "manual": manual_hub_selection,
    "mcs": mcs_star_map,
}

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate FEP edge map")
    parser.add_argument("config", help="YAML configuration")
    args = parser.parse_args()
    config = TimemachineConfig.from_yaml(args.config)
    if config.map_generation is None:
        print("No map generation configuration provided")
        sys.exit(1)
    map_config = config.map_generation

    with open(map_config.forcefield) as f:
        ff_handlers = deserialize_handlers(f.read())
    forcefield = Forcefield(ff_handlers)

    mols = []
    for lig_path in map_config.ligands:
        supplier = Chem.SDMolSupplier(lig_path, removeHs=False)
        mols.extend(list(supplier))

    # In the future hopefully we can programmatically find the cores rather specifying
    cores = map_config.cores

    core_sets = {}
    for i, core in enumerate(cores):
        method = core.pop("method")
        if method not in core_generation_methods:
            print(f"Unknown core method: {method}")
            sys.exit(1)
        core_method = core_generation_methods[method](**core)
        # Returns a match set and those that didn't match a core
        core_sets[i], mols = core_method(mols)

    if len(mols) > 1:
        print("Not all mols matched the provided cores")
        leftover = [get_mol_id(mol, map_config.identifier) for mol in mols if mol is not None]
        print(f"Mols that didn't match cores: {leftover}")

    all_edges = []
    for i, mols in core_sets.items():
        method = map_config.networks[i].pop("method")
        if method not in network_generation_methods:
            print(f"Unknown network method: {method}")
            sys.exit(1)
        hub_method = network_generation_methods[method](**map_config.networks[i])
        hub, mols = hub_method(mols)
        edges, errors = generate_star(hub, mols, forcefield, transformation_size_threshold=map_config.transformation_threshold, core_strategy=map_config.atom_mapping_strategy, label_property=map_config.label)
        all_edges.extend(edges)
        for hub, spoke, _ in errors:
            print(f'atom mapping error in transformation {get_mol_id(hub, map_config.identifier)} -> {get_mol_id(spoke, map_config.identifier)}!')
        with open(f"core_{i}_error_transformations.pkl", "wb") as f:
            dump(errors, f)
        print(f"Core {i} had {len(edges)} edges and {len(errors)} errors")

    # serialize
    with open(map_config.output, 'wb') as f:
        dump(all_edges, f)
