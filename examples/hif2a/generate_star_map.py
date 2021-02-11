# Construct a star map for the fep-benchmark hif2a ligands

from pathlib import Path
from rdkit import Chem

from pickle import dump

from fe import topology
from fe.utils import convert_uIC50_to_kJ_per_mole

root = Path(__file__).parent.parent.parent

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

# from fe.atom_mapping import get_core_by_geometry, get_core_by_matching, get_core_by_mcs
from fe.atom_mapping import get_core_by_smarts

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
mols_with_core_1 = [mol for mol in mols if len(mol.GetSubstructMatches(bicyclic_query_mol)) == 1]


# Draw.MolsToGridImage(mols_with_core_1, molsPerRow=5, subImgSize=(200,200))
# TODO: save this image...


def get_mol_id(mol):
    return mol.GetProp('ID')


# hard-coded hub = mol ID 165
mol_ids = list(map(get_mol_id, mols_with_core_1))
hub_index = mol_ids.index('165')

spokes = list(mols_with_core_1)
hub = spokes.pop(hub_index)

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


# note: some of transformations may fail the factorizability assertion here:
# https://github.com/proteneer/timemachine/blob/2eb956f9f8ce62287cc531188d1d1481832c5e96/fe/topology.py#L381-L431
transformations = []
error_transformations = []
for spoke in spokes:
    # core = get_core_by_mcs(hub, spoke, mcs_map(hub, spoke).queryMol)
    # core = get_core_by_matching(hub, spoke, threshold=0.9)
    # core = get_core_by_geometry(hub, spoke, threshold=0.5)
    core = get_core_by_smarts(hub, spoke, core_smarts=bicyclic_smarts_pattern)

    # TODO: reduce overlap between get_core_by_smarts and get_core_by_mcs
    # TODO: replace big ol' list of get_core_by_*(mol_a, mol_b, **kwargs) functions with something... classy

    try:
        single_topology = topology.SingleTopology(hub, spoke, core, forcefield)
        rfe = RelativeFreeEnergy(single_topology, label=_compute_label(hub, spoke))
        transformations.append(rfe)
    except AtomMappingError as e:
        print(f'atom mapping error in transformation {get_mol_id(hub)} -> {get_mol_id(spoke)}!')
        print(e)
        error_transformations.append((hub, spoke, core))

with open(path_to_results.joinpath('error_transformations.pkl'), 'wb') as f:
    dump(error_transformations, f)

print(f'total # of molecules that encountered atom mapping errors: {len(error_transformations)}')

# serialize

with open(path_to_transformations, 'wb') as f:
    dump(transformations, f)
