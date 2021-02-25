# Construct a star map for the fep-benchmark hif2a ligands

from pathlib import Path
from rdkit import Chem

from pickle import dump

from fe import topology
from fe.utils import convert_uIC50_to_kJ_per_mole

root = Path(__file__).parent.parent.parent

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

import numpy as np
from fe.atom_mapping import get_core_by_geometry
from fe.atom_mapping import get_core_by_matching, get_core_by_mcs, get_core_by_smarts, mcs_map

path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))
with open(path_to_ff) as f:
    ff_handlers = deserialize_handlers(f.read())
forcefield = Forcefield(ff_handlers)

# 1. Get ligands
path_to_ligands = str(root.joinpath('datasets/fep-benchmark/hif2a/ligands.sdf'))

# locations relative to example folder
path_to_results = Path(__file__).parent
path_to_transformations = str(path_to_results.joinpath('relative_transformations.pkl'))

supplier = Chem.SDMolSupplier(path_to_ligands, removeHs=False)
mols = []
for mol in supplier:
    mols.append(mol)

# 2. Identify ligand subset that shares a common substructure

# SMARTS courtesy of YTZ (Jan 26, 2021)
bicyclic_smarts_pattern = '[*]~1~[*]~[*]~2~[*]~[*]~[*]~[*](~[#8]~[*]~3~[*]~[*]~[*]~[*]~[*]~3)~[*]~2~[*]~1'
bicyclic_query_mol = Chem.MolFromSmarts(bicyclic_smarts_pattern)

# filter matches
has_bicycle = lambda mol: len(mol.GetSubstructMatches(bicyclic_query_mol)) == 1
mols_with_core_1 = list(filter(has_bicycle, mols))
mols_with_core_2 = list(filter(lambda mol: not has_bicycle(mol), mols))


# Draw.MolsToGridImage(mols_with_core_1, molsPerRow=5, subImgSize=(200,200))
# TODO: save this image...


def get_mol_id(mol):
    return mol.GetProp('ID')


from fe.free_energy import RelativeFreeEnergy
from fe.topology import AtomMappingError


def _compute_label(mol_a, mol_b):
    """ Compute labeled ddg (in kJ/mol) from the experimental IC50 s """

    prop_name = "IC50[uM](SPA)"
    try:
        label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp(prop_name)))
        label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp(prop_name)))
    except KeyError as e:
        raise (RuntimeError(f"Couldn't access IC50 label for either mol A or mol B, looking at {prop_name}"))

    label = label_dG_b - label_dG_a

    return label


# filter by transformation size
def transformation_size(rfe: RelativeFreeEnergy):
    n_A, n_B, n_MCS = rfe.mol_a.GetNumAtoms(), rfe.mol_b.GetNumAtoms(), len(rfe.core)
    return (n_A + n_B) - 2 * n_MCS


from rdkit.Chem import rdFMCS

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


#core_2_smarts = '[*]1~[*]~[*]~[*]~[*]~[*]1~[#8]~[*]2~[*]~[*]~[*](~[#16])~[*]~[*]2' # zero valid still!
# try removing any match in the sulfur-containing group?
core_2_smarts = '[*]1~[*]~[*]~[*]~[*]~[*]1~[#8]~[*]2~[*]~[*]~[*]~[*]~[*]2'

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


def generate_star(hub, spokes, transformation_size_threshold=2, core_strategy='geometry'):
    transformations = []
    error_transformations = []
    for spoke in spokes:
        core = core_strategies[core_strategy](hub, spoke)

        # TODO: reduce overlap between get_core_by_smarts and get_core_by_mcs
        # TODO: replace big ol' list of get_core_by_*(mol_a, mol_b, **kwargs) functions with something... classy

        try:
            single_topology = topology.SingleTopology(hub, spoke, core, forcefield)
            rfe = RelativeFreeEnergy(single_topology, label=_compute_label(hub, spoke))
            transformations.append(rfe)
        except AtomMappingError as e:
            # note: some of transformations may fail the factorizability assertion here:
            # https://github.com/proteneer/timemachine/blob/2eb956f9f8ce62287cc531188d1d1481832c5e96/fe/topology.py#L381-L431
            print(f'atom mapping error in transformation {get_mol_id(hub)} -> {get_mol_id(spoke)}!')
            print(e)
            error_transformations.append((hub, spoke, core))

    with open(path_to_results.joinpath('error_transformations.pkl'), 'wb') as f:
        dump(error_transformations, f)

    print(f'total # of edges that encountered atom mapping errors: {len(error_transformations)}')

    # filter to keep just the edges with very small number of atoms changing
    easy_transformations = [rfe for rfe in transformations if transformation_size(rfe) <= transformation_size_threshold]

    return easy_transformations


# add edges from each core

transformation_size_threshold = 3

# hard-coded hub = mol ID 165 for core 1
mol_inds_1 = list(map(get_mol_id, mols_with_core_1))
hub_index_1 = mol_inds_1.index('165')

spokes_1 = list(mols_with_core_1)
hub_1 = spokes_1.pop(hub_index_1)

core_1_edges = generate_star(hub_1, spokes_1, transformation_size_threshold)
print(f'# core 1 edges: {len(core_1_edges)}')

# add second core, filtered again by transformation difficulty

# hard-coded hub = mol ID 41
core_2_query_mol = Chem.MolFromSmarts(core_2_smarts)
has_core_2 = lambda mol: len(mol.GetSubstructMatches(core_2_query_mol)) > 0
mols_with_core_2 = list(filter(has_core_2, mols_with_core_2)) # filter outlier
mol_inds_2 = list(map(get_mol_id, mols_with_core_2))
hub_index_2 = mol_inds_2.index('41') # everything breaks with this as the hub, although it was identified as the best hub in the previous step

spokes_2 = list(mols_with_core_2)
hub_2 = spokes_2.pop(hub_index_2)

core_2_edges_dict = dict()
core_2_strategies = ['geometry', 'any_mcs', 'smarts_core_2']

for strategy in core_2_strategies:
    core_2_edges_dict[strategy] = generate_star(hub_2, spokes_2, transformation_size_threshold, core_strategy=strategy)

# see what gets the most edges that pass tests...
max_edges = 0
best_core_2_edges = core_2_edges_dict[core_2_strategies[0]]
for strategy in core_2_strategies:
    n_edges = len(core_2_edges_dict[strategy])
    print(f'# core 2 edges using {strategy} strategy: {n_edges}')
    if n_edges > max_edges:
        max_edges = n_edges
        best_core_2_edges = core_2_edges_dict[strategy]

# consistency with core 1
core_2_edges = core_2_edges_dict['geometry']


# [incomplete] let's maybe also try looping over all choices of possible hub, and see which gives us the most valid edges...
#for i in range(len(mols_with_core_2)):
#    spokes_2 = list(mols_with_core_2)
#    hub_2 = spokes_1.pop(i)
#    core_2_edges = generate_star(hub_2, spokes_2, transformation_size_threshold, core_strategy='smarts_core_2')


# combine
all_edges = core_1_edges + core_2_edges

# serialize
with open(path_to_transformations, 'wb') as f:
    dump(all_edges, f)
