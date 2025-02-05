from importlib import resources

import pytest
from openmm import app
from openmm.app.forcefield import ForceField as OMMForceField
from rdkit import Chem

from timemachine.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF
from timemachine.ff.handlers import utils as handler_utils
from timemachine.md import builders

pytestmark = [pytest.mark.nocuda]


def test_get_query_mol():
    mol_a = Chem.MolFromSmiles("NCC(O)=O")

    # with bond ordered only have one match
    mol_q = Chem.MolFromSmiles("C=O")
    matches = mol_a.GetSubstructMatches(mol_q)
    assert len(matches) == 1

    # using generic bonds, have two matches
    mol_q = handler_utils.get_query_mol(mol_q)
    matches = mol_a.GetSubstructMatches(mol_q)
    assert len(matches) == 2


@pytest.fixture()
def residue_mol_inputs():
    properties_by_res_name = {}
    for protein_path in ["5dfr_solv_equil.pdb", "hif2a_nowater_min.pdb"]:
        with resources.path("timemachine.testsystems.data", protein_path) as path_to_pdb:
            host_pdb = app.PDBFile(str(path_to_pdb))
            host_config = builders.build_protein_system(host_pdb, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)
        ff = OMMForceField(f"{DEFAULT_PROTEIN_FF}.xml", f"{DEFAULT_WATER_FF}.xml")
        data = OMMForceField._SystemData(host_config.omm_topology)
        residueTemplates = {}
        template_for_residue = ff._matchAllResiduesToTemplates(
            data, host_config.omm_topology, residueTemplates, ignoreExternalBonds=False
        )

        for tfr in template_for_residue:
            if tfr.name in properties_by_res_name:
                continue
            elements = [atom.element.symbol for atom in tfr.atoms]
            bond_list = tfr.bonds
            properties_by_res_name[tfr.name] = {
                "elements": elements,
                "bond_list": bond_list,
            }

    return properties_by_res_name


def test_make_residue_mol(residue_mol_inputs):
    for res_name, props in residue_mol_inputs.items():
        res_mol = handler_utils.make_residue_mol(res_name, props["elements"], props["bond_list"])
        assert res_mol.GetNumAtoms() == len(props["elements"])

        bond_list_mol = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in res_mol.GetBonds()]
        assert set(bond_list_mol) == set(props["bond_list"])


def test_make_residue_mol_from_template():
    for res_name in handler_utils.SMILES_BY_RES_NAME:
        res_mol = handler_utils.make_residue_mol_from_template(res_name)

        if res_name not in ["ACE", "NME"]:
            nres_mol = handler_utils.make_residue_mol_from_template(f"N{res_name}")
            cres_mol = handler_utils.make_residue_mol_from_template(f"C{res_name}")

            assert res_mol.GetNumAtoms() + 1 == nres_mol.GetNumAtoms()
            assert res_mol.GetNumAtoms() - 1 == cres_mol.GetNumAtoms()
        else:
            assert res_mol is not None

    assert handler_utils.make_residue_mol_from_template("HOH") is None


def test_update_mol_topology(residue_mol_inputs):
    for res_name, props in residue_mol_inputs.items():
        if res_name in ["HOH"]:  # doesn't apply
            continue
        elements, bond_list = props["elements"], props["bond_list"]
        topology_res_mol = handler_utils.make_residue_mol(res_name, elements, bond_list)
        template_res_mol = handler_utils.make_residue_mol_from_template(res_name)

        # from rdkit.Chem.Draw import rdMolDraw2D
        # with open(f"test.svg", "w") as fh:
        #     drawer = rdMolDraw2D.MolDraw2DSVG(200, 180 * 2, 200, 180)
        #     drawer.drawOptions().useBWAtomPalette()
        #     dopts = drawer.drawOptions()
        #     dopts.baseFontSize = 0.3

        #     drawer.DrawMolecules(
        #         [topology_res_mol, template_res_mol],
        #         legends=[res_name, res_name + "template"],
        #     )
        #     drawer.FinishDrawing()
        #     fh.write(drawer.GetDrawingText())

        handler_utils.update_mol_topology(topology_res_mol, template_res_mol)

        match = template_res_mol.GetSubstructMatch(topology_res_mol)

        # map from topology_res_mol to template_res_mol
        fwd_map = {i: v for i, v in enumerate(match)}

        # all atoms should have the same formal charge
        for topology_atom in topology_res_mol.GetAtoms():
            template_atom = template_res_mol.GetAtomWithIdx(fwd_map[topology_atom.GetIdx()])
            assert template_atom.GetFormalCharge() == topology_atom.GetFormalCharge()

        # all bonds should have the same bond type/aromaticity
        template_bonds = {}
        for bond in template_res_mol.GetBonds():
            src_idx = bond.GetBeginAtomIdx()
            dst_idx = bond.GetEndAtomIdx()
            template_bonds[(src_idx, dst_idx)] = bond
            template_bonds[(dst_idx, src_idx)] = bond

        for topology_bond in topology_res_mol.GetBonds():
            src_idx = fwd_map[topology_bond.GetBeginAtomIdx()]
            dst_idx = fwd_map[topology_bond.GetEndAtomIdx()]
            template_bond = template_bonds[(src_idx, dst_idx)]
            assert topology_bond.GetBondType() == template_bond.GetBondType()
            assert topology_bond.GetIsAromatic() == template_bond.GetIsAromatic()


def test_get_res_name():
    res_name, has_n_cap, has_c_cap = handler_utils.get_res_name("ARG")
    assert res_name == "ARG"
    assert not has_n_cap
    assert not has_c_cap

    res_name, has_n_cap, has_c_cap = handler_utils.get_res_name("NARG")
    assert res_name == "ARG"
    assert has_n_cap
    assert not has_c_cap

    res_name, has_n_cap, has_c_cap = handler_utils.get_res_name("CARG")
    assert res_name == "ARG"
    assert not has_n_cap
    assert has_c_cap


def test_add_n_cap():
    for res_name in handler_utils.SMILES_BY_RES_NAME:
        if res_name in ["ACE", "NME"]:  # already capped
            continue
        res_mol = handler_utils.make_residue_mol_from_template(res_name)
        res_mol = handler_utils.add_n_cap(res_mol)

        params = Chem.SmilesParserParams()
        params.removeHs = False
        if res_name == "PRO":
            query_mol = Chem.MolFromSmiles("[H]C([H])[N+]([H])([H])[C@@]([H])(C([H])[H])C(O)=O", params)
        else:
            query_mol = Chem.MolFromSmiles("[H]C(C(O)=O)[N+]([H])([H])[H]", params)

        matches = res_mol.GetSubstructMatches(query_mol)
        if res_name == "GLY":
            assert len(matches) == 2  # for matching either 'H'
        else:
            assert len(matches) == 1


def test_add_c_cap():
    for res_name in handler_utils.SMILES_BY_RES_NAME:
        if res_name in ["ACE", "NME"]:  # already capped
            continue
        res_mol = handler_utils.make_residue_mol_from_template(res_name)
        res_mol = handler_utils.add_c_cap(res_mol)
        matches = res_mol.GetSubstructMatches(Chem.MolFromSmiles("NCC([O-])=O"))
        assert len(matches) == 1
