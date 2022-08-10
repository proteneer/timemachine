import argparse
import csv
import pickle

import numpy as np
from rdkit import Chem

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from timemachine.constants import KCAL_TO_KJ
from timemachine.fe import atom_mapping
from timemachine.fe.rbfe import SimulationException, run_pair
from timemachine.fe.utils import get_mol_name
from timemachine.parallel.client import CUDAPoolClient


def get_mol_by_name(mols, name):
    for m in mols:
        if get_mol_name(m) == name:
            return m

    assert 0, "Mol not found"


def read_from_args():

    parser = argparse.ArgumentParser(
        description="Estimate relative free energy difference between complex and solvent given a results csv file."
    )
    parser.add_argument(
        "--n_frames", type=int, help="number of frames to use for the free energy estimate", required=True
    )
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--results_csv", type=str, help="Results containing all the csv", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", required=True)
    parser.add_argument("--protein", type=str, help="PDB of the protein complex", required=True)
    parser.add_argument("--n_gpus", type=int, help="number of gpus", required=True)
    parser.add_argument("--seed", type=int, help="random seed for the runs", required=True)

    args = parser.parse_args()

    mols = [mol for mol in Chem.SDMolSupplier(str(args.ligands), removeHs=False)]

    futures = []
    metadata = []

    cpc = CUDAPoolClient(args.n_gpus)
    cpc.verify()

    with open(args.results_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        rows = [row for row in reader]
        for row_idx, row in enumerate(rows):
            mol_a_name, mol_b_name, exp_ddg, fep_ddg, fep_ddg_err, ccc_ddg, ccc_ddg_err = row
            mol_a = get_mol_by_name(mols, mol_a_name)
            mol_b = get_mol_by_name(mols, mol_b_name)

            print(f"Submitting job for {mol_a_name} -> {mol_b_name}")
            mcs_threshold = 2.0
            mcs_result = atom_mapping.mcs_map(mol_a, mol_b, threshold=mcs_threshold)
            query_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            core = atom_mapping.get_core_by_mcs(mol_a, mol_b, query_mol, threshold=mcs_threshold)
            fut = cpc.submit(
                run_pair, mol_a, mol_b, core, args.forcefield, args.protein, args.n_frames, args.seed + row_idx
            )
            futures.append(fut)

            metadata.append(
                (
                    mol_a,
                    mol_b,
                    mcs_result.smartsString,
                    core,
                    float(exp_ddg) * KCAL_TO_KJ,
                    float(fep_ddg) * KCAL_TO_KJ,
                    float(fep_ddg_err) * KCAL_TO_KJ,
                    float(ccc_ddg) * KCAL_TO_KJ,
                    float(ccc_ddg_err) * KCAL_TO_KJ,
                )
            )

    for fut, meta in zip(futures, metadata):

        mol_a, mol_b, _, _, exp_ddg, fep_ddg, fep_ddg_err, ccc_ddg, ccc_ddg_err = meta
        mol_a_name = get_mol_name(mol_a)
        mol_b_name = get_mol_name(mol_b)

        try:
            solvent_res, solvent_top, complex_res, complex_top = fut.result()

            with open(f"success_rbfe_result_{mol_a_name}_{mol_b_name}.pkl", "wb") as fh:
                pkl_obj = (meta, solvent_res, solvent_top, complex_res, complex_top)
                pickle.dump(pkl_obj, fh)

            solvent_ddg = np.sum(solvent_res.all_dGs)
            solvent_ddg_err = np.linalg.norm(solvent_res.all_errs)
            complex_ddg = np.sum(complex_res.all_dGs)
            complex_ddg_err = np.linalg.norm(complex_res.all_errs)

            tm_ddg = complex_ddg - solvent_ddg
            tm_err = np.linalg.norm([complex_ddg_err, solvent_ddg_err])

            print(
                f"finished: {mol_a_name} -> {mol_b_name} (kJ/mol) | complex {complex_ddg:.2f} +- {complex_ddg_err:.2f} | solvent {solvent_ddg:.2f} +- {solvent_ddg_err:.2f} | tm_pred {tm_ddg:.2f} +- {tm_err:.2f} | exp_ddg {exp_ddg:.2f} | fep_ddg {fep_ddg:.2f} +- {fep_ddg_err:.2f}"
            )

        except SimulationException as sim_exc:

            print("Failed", sim_exc)
            with open(f"failed_rbfe_result_{mol_a_name}_{mol_b_name}.pkl", "wb") as fh:
                pickle.dump(sim_exc, fh)


if __name__ == "__main__":

    read_from_args()
