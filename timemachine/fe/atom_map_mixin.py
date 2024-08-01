from enum import IntEnum

import numpy as np


class AtomMapFlags(IntEnum):
    CORE = 0
    MOL_A = 1
    MOL_B = 2


class AtomMapMixin:
    """
    A Mixin class containing the atom_mapping information. This Mixin sets up the following
    members:

    self.mol_a
    self.mol_b
    self.core
    self.a_to_c
    self.b_to_c
    self.c_to_a
    self.c_to_b
    self.c_flags
    """

    def __init__(self, mol_a, mol_b, core):
        assert core.shape[1] == 2
        assert mol_a is not None
        assert mol_b is not None

        self.mol_a = mol_a
        self.mol_b = mol_b
        self.core = core
        assert mol_a is not None
        assert mol_b is not None
        assert core.shape[1] == 2

        # map into idxs in the combined molecule

        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32)  # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        # mark membership:
        # AtomMapFlags.CORE: Core
        # AtomMapFlags.MOL_A: R_A (default)
        # AtomMapFlags.MOL_B: R_B
        self.c_flags = np.ones(self.get_num_atoms(), dtype=np.int32) * AtomMapFlags.MOL_A
        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

        for a, b in core:
            self.c_flags[a] = AtomMapFlags.CORE
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = AtomMapFlags.MOL_B
                iota += 1

        # setup reverse mappings
        self.c_to_a = {v: k for k, v in enumerate(self.a_to_c)}
        self.c_to_b = {v: k for k, v in enumerate(self.b_to_c)}

    def get_num_atoms(self):
        """
        Get the total number of atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core)

    def get_num_dummy_atoms(self):
        """
        Get the total number of dummy atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core) - len(self.core)
