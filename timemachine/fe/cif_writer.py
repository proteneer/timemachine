import numpy as np
from openmm import app
from openmm.app import PDBxFile
from rdkit import Chem

from timemachine.fe.single_topology import AtomMapMixin


def convert_single_topology_mols(coords: np.ndarray, atom_map: AtomMapMixin) -> np.ndarray:
    """Convert a single topology frame's ligand coordinates into two complete ligands.

    coords: np.ndarray
        The coordinates that represent the single topology alchemical ligand

    atom_map: AtomMapMixin or SingleTopology
        Mixin containing the atom_mapping information


    Example
    -------

        >>> writer = CIFWriter(...)
        >>> lig_coords = convert_single_topology_mols(x0[num_host_coords:], atom_map)
        >>> new_coords = np.concatenate((x0[:num_host_coords], lig_coords), axis=0)
        >>> writer.write_frame(new_coords*10)
        >>> writer.close()

    """
    xa = np.zeros((atom_map.num_atoms_a, 3))
    xb = np.zeros((atom_map.num_atoms_b, 3))
    for a_idx, c_idx in enumerate(atom_map.a_to_c):
        xa[a_idx] = coords[c_idx]
    for b_idx, c_idx in enumerate(atom_map.b_to_c):
        xb[b_idx] = coords[c_idx]
    return np.concatenate((xa, xb), axis=0)


class BondTypeError(Exception):
    pass


class CIFWriter:
    def __init__(self, objs, out_filepath):
        """
        This class writes frames out in the mmCIF format. It supports both OpenMM topology
        formats and RDKit ROMol types. The molecules are ordered sequentially by the order
        in which they are in objs

        Usage:

        ```
        topol = app.PDBFile("1dfr.pdb").topology
        mol_a = Chem.MolFromMolBlock("...")
        mol_b = Chem.MolFromMolBlock("...")

        writer = CIFWriter([topol, mol_a, mol_b], out.cif) # writer header
        writer.write_frame(coords) # coords must be in units of angstroms
        writer.close() # writes a hashtag
        ```

        Parameters
        ----------
        objs: list of either Chem.Mol or app.Topology types
            Molecules of interest

        out_filepath: str
            Where we write out the CIF file to

        """

        assert len(objs) > 0

        combined_topology = app.Topology()

        used_residue_ids = []

        # see if an existing topology is present
        for obj_idx, obj in enumerate(objs):
            old_topology = obj
            if isinstance(obj, app.Topology):
                old_chain_obj_kv = {}

                for old_chain in old_topology.chains():
                    new_chain = combined_topology.addChain()
                    old_chain_obj_kv[old_chain] = new_chain

                old_atom_obj_kv = {}
                for old_residue in old_topology.residues():
                    chain_obj = old_chain_obj_kv[old_residue.chain]
                    new_residue = combined_topology.addResidue(
                        name=old_residue.name,
                        chain=chain_obj,
                        id=(old_residue.id + old_residue.insertionCode if old_residue.insertionCode else "")
                        if old_residue.name != "HOH"
                        else str(max(used_residue_ids) + 1),
                    )

                    try:
                        used_residue_ids.append(int(new_residue.id))
                    except ValueError:
                        # not an int id
                        pass
                    for old_atom in old_residue.atoms():
                        new_atom = combined_topology.addAtom(old_atom.name, old_atom.element, new_residue)
                        assert old_atom not in old_atom_obj_kv
                        old_atom_obj_kv[old_atom] = new_atom

                for old_bond in old_topology.bonds():
                    new_atom1 = old_atom_obj_kv[old_bond.atom1]
                    new_atom2 = old_atom_obj_kv[old_bond.atom2]
                    combined_topology.addBond(new_atom1, new_atom2, old_bond.type, old_bond.order)

                # assert that we've added everything
                assert len(old_atom_obj_kv) == obj.getNumAtoms()

            elif isinstance(obj, Chem.Mol):
                mol = obj
                new_chain = combined_topology.addChain()
                new_residue = combined_topology.addResidue(
                    name="LIG", chain=new_chain, id=str(max(used_residue_ids) + 1)
                )

                try:
                    used_residue_ids.append(int(new_residue.id))
                except ValueError:
                    # not an int id
                    pass
                old_idx_to_new_atom_map = {}
                for atom in mol.GetAtoms():
                    name = atom.GetSymbol() + str(atom.GetIdx())
                    element = app.element.Element.getByAtomicNumber(atom.GetAtomicNum())
                    new_atom = combined_topology.addAtom(name, element, new_residue)
                    old_idx = atom.GetIdx()
                    old_idx_to_new_atom_map[old_idx] = new_atom

                # (ytz): while we could get fancier, mmcif only has the 'covale' bond type:
                # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Items/_struct_conn_type.id.html
                # so we only bother with "yes/no"
                for bond in mol.GetBonds():
                    src_atom = old_idx_to_new_atom_map[bond.GetBeginAtomIdx()]
                    dst_atom = old_idx_to_new_atom_map[bond.GetEndAtomIdx()]
                    combined_topology.addBond(src_atom, dst_atom)

            else:
                raise ValueError(f"Unknown obj type: {type(obj)}")

        # assert that ids are unique.
        atom_ids = list([x.id for x in combined_topology.atoms()])
        assert len(atom_ids) == len(set(atom_ids))

        residue_ids = list([x.id for x in combined_topology.residues()])
        assert len(residue_ids) == len(set(residue_ids))

        self.topology = combined_topology
        self.out_handle = open(out_filepath, "w")
        PDBxFile.writeHeader(self.topology, self.out_handle, keepIds=True)
        self.topology = self.topology
        self.frame_idx = 0

    def write_frame(self, x):
        """
        Write a coordinate frame

        Parameters
        ----------
        x: np.ndarray
            coordinates in units of angstroms

        """
        self.frame_idx += 1
        PDBxFile.writeModel(self.topology, x, self.out_handle, self.frame_idx, keepIds=True)

    def close(self):
        # Need this final #
        self.out_handle.write("#")
        self.out_handle.flush()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
