from jax.config import config

config.update("jax_enable_x64", True)

import functools
from importlib import resources

import numpy as np
import pytest
from common import GradientTest
from rdkit import Chem

from timemachine.lib import potentials
from timemachine.potentials import shape


def recenter(conf):
    """Return copy of conf with mean coordinates subtracted"""
    return conf - np.mean(conf, axis=0)


def get_conf(romol, idx):
    """Get the idx'th conformer of romol, in nanometers"""
    conformer = romol.GetConformer(idx)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf /= 10  # TODO: label this unit conversion?
    return recenter(guest_conf)


def make_conformer(mol, conf_a, conf_b):
    """Remove all of mol's conformers, make a new mol containing two copies of mol,
    assign positions to each copy using conf_a and conf_b, respectively, assumed in nanometers"""
    mol.RemoveAllConformers()
    mol = Chem.CombineMols(mol, mol)
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.concatenate([conf_a, conf_b])
    conf *= 10  # TODO: label this unit conversion?
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


def get_heavy_atom_idxs(mol):
    """Return integer array of indices of non-hydrogen's"""
    idxs = []
    for a_idx, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() > 1:
            idxs.append(a_idx)
    return np.array(idxs, dtype=np.int32)


class TestShape(GradientTest):
    def test_volume_range(self):
        """Assert that:
        * normalized overlap for identical configurations = 1
        * normalized overlap for similar configurations is between 0.5 and 1.0
        """

        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

        prefactor = 2.7  # unitless
        lamb = (4 * np.pi) / (3 * prefactor)  # unitless
        kappa = np.pi / (np.power(lamb, 2 / 3))  # unitless
        sigma = 1.6  # angstroms or nm # TODO: check unit?
        alpha = kappa / (sigma * sigma)

        for ligand_a in suppl:

            coords_a = get_conf(ligand_a, idx=0) * 10
            params_a = np.stack(
                [
                    np.zeros(ligand_a.GetNumAtoms()) + alpha,
                    np.zeros(ligand_a.GetNumAtoms()) + prefactor,
                ],
                axis=1,
            )

            v = shape.normalized_overlap(coords_a, params_a, coords_a, params_a)

            assert v == 1.0

            for ligand_b in suppl:
                coords_b = get_conf(ligand_b, idx=0) * 10
                params_b = np.stack(
                    [
                        np.zeros(ligand_b.GetNumAtoms()) + alpha,
                        np.zeros(ligand_b.GetNumAtoms()) + prefactor,
                    ],
                    axis=1,
                )

                v = shape.normalized_overlap(coords_a, params_a, coords_b, params_b)

                assert v <= 1
                assert v >= 0.5

    @pytest.mark.skip("disabled")
    def test_custom_op(self):
        """Construct a Shape potential and validate it using compare_forces"""
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

        prefactor = 2.7  # unitless
        lamb = (4 * np.pi) / (3 * prefactor)  # unitless
        kappa = np.pi / (np.power(lamb, 2 / 3))  # unitless
        sigma = 1.6  # angstroms or nm # TODO: check unit?
        alpha = kappa / (sigma * sigma)
        # TODO: extract the preceding setup block and share between test_custom_op and test_volume_range?

        for ligand_a in suppl:

            coords_a = get_conf(ligand_a, idx=0) * 10

            a_alphas = np.random.rand(ligand_a.GetNumAtoms()) / 10 + alpha
            a_weights = np.random.rand(ligand_a.GetNumAtoms()) / 10 + prefactor
            a_idxs = get_heavy_atom_idxs(ligand_a)

            for j_idx, ligand_b in enumerate(suppl):

                b_alphas = np.random.rand(ligand_b.GetNumAtoms()) / 10 + alpha
                b_weights = np.random.rand(ligand_b.GetNumAtoms()) / 10 + prefactor

                coords_b = get_conf(ligand_b, idx=0) * 10
                coords = np.concatenate([coords_a, coords_b])
                b_idxs = get_heavy_atom_idxs(ligand_b)
                b_idxs += ligand_a.GetNumAtoms()

                c_alphas = np.concatenate([a_alphas, b_alphas])
                c_weights = np.concatenate([a_weights, b_weights])

                k = 195.0

                ref_u = functools.partial(
                    shape.harmonic_overlap, alphas=c_alphas, weights=c_weights, a_idxs=a_idxs, b_idxs=b_idxs, k=k
                )

                test_u = potentials.Shape(
                    coords.shape[0],
                    a_idxs.astype(np.int32),
                    b_idxs.astype(np.int32),
                    c_alphas.astype(np.float64),
                    c_weights.astype(np.float64),
                    k,
                )

                # for identical molecules 32 bit precision is terrible since the overlap is super small
                for rtol, precision in [(1e-4, np.float32), (1e-10, np.float64)]:
                    self.compare_forces(
                        x=coords,
                        params=np.array([]),
                        box=np.eye(3) * 1000,
                        lamb=0.0,
                        ref_potential=ref_u,
                        test_potential=test_u,
                        rtol=rtol,
                        precision=precision,
                    )

                # assert 0
