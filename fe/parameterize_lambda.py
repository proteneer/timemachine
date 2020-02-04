import argparse
import time
import numpy as np
from io import StringIO
import itertools
import gnuplotlib as gp
import os
import sys

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)


from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils

import multiprocessing
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation
from fe import loss


def com(conf):
    return np.sum(conf, axis=0)/conf.shape[0]

def recenter(conf, true_com, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 

from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_sort(conf):
    hc = HilbertCurve(16, 3)
    int_confs = (conf*1000).astype(np.int64)+10000
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm

class PDBWriter():

    def __init__(self, pdb_str, out_filepath):
        self.pdb_str = pdb_str
        self.out_filepath = out_filepath
        self.outfile = None
        self.n_frames = 100

    def write_header(self):
        """
        Confusingly this initializes writer as well because 
        """
        outfile = open(self.out_filepath, 'w')
        self.outfile = outfile
        cpdb = app.PDBFile(self.pdb_str)
        PDBFile.writeHeader(cpdb.topology, self.outfile)
        self.topology = cpdb.topology
        self.frame_idx = 0

    def write(self, x):
        if self.outfile is None:
            raise ValueError("remember to call write_header first")
        self.frame_idx += 1
        PDBFile.writeModel(self.topology, x, self.outfile, self.frame_idx)

    def close(self):
        PDBFile.writeFooter(self.topology, self.outfile)
        self.outfile.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_pdb', type=str)

    # parser.add_argument('--precision', type=str)    
    parser.add_argument('--complex_pdb', type=str)
    parser.add_argument('--solvent_pdb', type=str)
    parser.add_argument('--ligand_sdf', type=str)
    args = parser.parse_args()

    amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')

    # host_pdb = app.PDBFile(args.complex_pdb)
    # modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    # modeller.addSolvent(amber_ff, numAdded=6000)
    # PDBFile.writeFile(modeller.topology, modeller.positions, open("sanitized.pdb", 'w'))
    # assert 0

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    for guest_mol in suppl:
        break

    T = 10000
    dt = 0.0015
    step_sizes = np.ones(T)*dt
    cas = np.ones(T)*0.99

    num_gpus = 1
    pool = multiprocessing.Pool(num_gpus)

    all_du_dls = []

    lambda_schedule = np.linspace(0.00001, 0.99999, num=T)

    epoch = 0

    init_conf = guest_mol.GetConformer(0)
    init_conf = np.array(init_conf.GetPositions(), dtype=np.float64)
    init_conf = init_conf/10 # convert to md_units
    conf_com = com(init_conf)
    init_mol = Chem.Mol(guest_mol)

    for host_idx, host_pdb_file in enumerate([args.complex_pdb, args.solvent_pdb]):

        host_pdb = app.PDBFile(host_pdb_file)

        if host_idx == 0:
            host_name = "complex"
        elif host_idx == 1:
            host_name = "solvent"

        for mode in ['insertion', 'deletion']:
            print("Mode", mode)

            host_conf = []
            for x,y,z in host_pdb.positions:
                host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])

            host_conf = np.array(host_conf)
            init_combined_conf = np.concatenate([host_conf, init_conf])

            perm = hilbert_sort(init_combined_conf)
            sim = simulation.Simulation(
                guest_mol,
                host_pdb,
                mode,
                step_sizes,
                cas,
                lambda_schedule,
                perm
            )

            num_conformers = 1

            guest_mol.RemoveAllConformers()
            AllChem.EmbedMultipleConfs(guest_mol, num_conformers, randomSeed=2020)

            # sample from the DG distribution
            all_args = []
            for conf_idx in range(num_conformers):
                guest_conf = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(guest_conf.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                np.random.seed(2020)
                rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
                guest_conf = np.matmul(guest_conf, rot_matrix)
                guest_conf = recenter(guest_conf, conf_com)

                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry
                x0 = x0[perm]

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join("frames", "epoch_"+str(epoch)+"_"+mode+"_"+host_name+"_conf_"+str(conf_idx)+".pdb")
                writer = PDBWriter(combined_pdb_str, out_file)
                all_args.append((x0, writer,  conf_idx % num_gpus))

            sim.run_forward_multi(all_args[0])

            sys.exit(0)

            results = pool.map(sim.run_forward_multi, all_args)
            all_du_dls.append(results)

    all_du_dls = np.array(all_du_dls)

    error = loss.loss_dG(*all_du_dls, lambda_schedule, 250.)

    print("Error", error)

    assert 0
