import hashlib

from rdkit import Chem

from timemachine.fe import functional


def bind_potentials(potentials, params):
    """modifies potentials in-place"""
    for U, p in zip(potentials, params):
        U.bind(p)


def construct_potential(potentials, params):
    U_fn = functional.construct_differentiable_interface_fast(potentials, params)

    def potential(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam)

    return potential


def hash_mol(mol: Chem.Mol) -> str:
    # TODO: check if there's a more idiomatic way to do this
    canon_smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    mol_hash = hashlib.sha256(canon_smiles.encode("utf-8")).hexdigest()
    short_version = mol_hash[:7]
    return short_version
