import numpy as np

import tempfile
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Topology

# class PDBWriter():

#     def __init__(self, pdb_str, out_filepath):
#         self.pdb_str = pdb_str
#         self.out_filepath = out_filepath
#         self.outfile = None

#     def write_header(self, box=None):
#         """
#         Confusingly this initializes writer as well because 
#         """
#         outfile = open(self.out_filepath, 'w')
#         self.outfile = outfile
#         cpdb = app.PDBFile(self.pdb_str)
#         if box is not None:
#             cpdb.topology.setPeriodicBoxVectors(box)
#         PDBFile.writeHeader(cpdb.topology, self.outfile)
#         self.topology = cpdb.topology
#         self.frame_idx = 0

#     def write(self, x):
#         if self.outfile is None:
#             raise ValueError("remember to call write_header first")
#         self.frame_idx += 1
    
#         PDBFile.writeModel(self.topology, x, self.outfile, self.frame_idx)

#     def close(self):
#         PDBFile.writeFooter(self.topology, self.outfile)
#         self.outfile.flush()


from rdkit import Chem

class PDBWriter():

    def __init__(self, objs, out_filepath): 
        """
        This class writes frames out in the PDBFormat. It supports both OpenMM topology
        formats and RDKit ROMol types. The molecules are ordered sequentially by the order
        in which they are in objs

        Usage:

        ```
        topol = app.PDBFile("1dfr.pdb").topology
        mol_a = Chem.MolFromMolBlock("...")
        mol_b = Chem.MolFromMolBlock("...")

        writer = PDBWriter([topol, mol_a, mol_b], out.pdb) # writer header
        writer.write_frame(coords) # coords must be in units of angstroms
        writer.close() # writes footer
        ```

        Parameters
        ----------
        obj: list of either Chem.Mol or app.Topoolgy types
            Molecules of interest

        out_filepath: str
            Where we write out the PDBFile to

        """

        assert len(objs) > 0

        rd_mols = []
        # super jank
        for obj in objs:
            if isinstance(obj, app.Topology):
                with tempfile.NamedTemporaryFile(mode='w') as fp:
                    # write
                    PDBFile.writeHeader(obj, fp)
                    PDBFile.writeModel(obj, np.zeros((obj.getNumAtoms(), 3)), fp, 0)
                    PDBFile.writeFooter(obj, fp)
                    fp.flush()
                    # read
                    rd_mols.append(Chem.MolFromPDBFile(fp.name, removeHs=False))

            if isinstance(obj, Chem.Mol):
                rd_mols.append(obj)

        combined_mol = rd_mols[0]
        for mol in rd_mols[1:]:
            combined_mol = Chem.CombineMols(combined_mol, mol)

        with tempfile.NamedTemporaryFile(mode='w') as fp:
            # write
            Chem.MolToPDBFile(combined_mol, fp.name)
            fp.flush()
            # read
            combined_pdb = app.PDBFile(fp.name)
            self.topology = combined_pdb.topology

        self.out_handle = open(out_filepath, "w")
        PDBFile.writeHeader(self.topology, self.out_handle)
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
        # if self.outfile is None:
            # raise ValueError("remember to call write_header first")
        self.frame_idx += 1
        PDBFile.writeModel(self.topology, x, self.out_handle, self.frame_idx)

    def close(self):
        """
        Write footer and close.
        """
        PDBFile.writeFooter(self.topology, self.out_handle)
        self.out_handle.flush()