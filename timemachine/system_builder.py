
from timemachine.cpu_functionals import custom_ops
import numpy as np

from openforcefield.typing.engines.smirnoff import forcefield

def convert_rmin_half_to_sigma(rmh):
    # rmh =  2^(1/6)*sigma/2
    return 2 * rmh / np.power(2, 1/6)


def addExclusionsToSet(bonded12, exclusions, baseParticle, fromParticle, currentLevel):
    for i in bonded12[fromParticle]:
        if i != baseParticle:
            exclusions.add(i)
        if currentLevel > 0:
            addExclusionsToSet(bonded12, exclusions, baseParticle, i, currentLevel-1)


def generate_scale_matrix(bond_idxs, scale, num_atoms):
    """
    Generate matrices that scales nonbonded interacitons.

    Parameters
    ----------
    np.array: shape (B,2)
        Bond indices

    scale: np.float32
        How much we should scale 14 interactions by.

    num_atoms: int
        Number of atoms in the system

    """

    exclusions = []
    bonded12 = []

    scale_matrix = np.ones((num_atoms, num_atoms), dtype=np.float32)
    # diagonals
    for a_idx in range(num_atoms):
        scale_matrix[a_idx][a_idx] = 0.0
        exclusions.append(set())
        bonded12.append(set())

    # taken from openmm's createExceptionsFromBonds()
    for first, second in bond_idxs:
        bonded12[first].add(second)
        bonded12[second].add(first)

    for i in range(num_atoms):
        addExclusionsToSet(bonded12, exclusions[i], i, i, 2)

    for i in range(num_atoms):
        bonded13 = set()
        addExclusionsToSet(bonded12, bonded13, i, i, 1)
        for j in exclusions[i]:
            if j < i:
                if j not in bonded13:
                    scale_matrix[i][j] = scale
                    scale_matrix[j][i] = scale
                else:
                    scale_matrix[i][j] = 0.0
                    scale_matrix[j][i] = 0.0

    return scale_matrix

def construct_energies(ff, mol):
    """
    Construct energies given a forcefield and a molecule.

    Parameters
    ----------
    ff: openforcefield.ForceField
        Pre-loaded forcefield.

    mol: oechem.OEMol
        OpenEye Mol object

    Returns
    -------
    list of energy object

    """
    labels = ff.labelMolecules([mol], verbose=True)
    N = mol.NumAtoms()
    start_params = 0
    nrgs = []
    for mol_entry in range(len(labels)):

        for force in labels[mol_entry].keys():
            if force == 'HarmonicBondGenerator':
                bond_params_map = {}
                bond_params_array = []
                bond_params_idxs = []
                bond_atom_idxs = []
                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    if pid not in bond_params_map:
                        params = ff.getParameter(paramID=pid)
                        k = np.float32(params['k'])
                        k_idx = len(bond_params_array)
                        bond_params_array.append(k)

                        length = np.float32(params['length'])
                        length_idx = len(bond_params_array)
                        bond_params_array.append(length)
                        bond_params_map[pid] = (k_idx, length_idx)

                    bond_params_idxs.extend(bond_params_map[pid])
                    bond_atom_idxs.extend(atom_indices)

                bond_nrg = custom_ops.HarmonicBondGPU_float(
                    bond_params_array,
                    list(range(start_params, start_params+len(bond_params_array))),
                    bond_params_idxs,
                    bond_atom_idxs
                )

                nrgs.append(bond_nrg)

                start_params += len(bond_params_array)
            elif force == 'HarmonicAngleGenerator':

                angle_params_map = {}
                angle_params_array = []
                angle_params_idxs = []
                angle_atom_idxs = []

                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    if pid not in angle_params_map:
                        params = ff.getParameter(paramID=pid)
                        k = np.float32(params['k'])
                        k_idx = len(angle_params_array)
                        angle_params_array.append(k)

                        angle = np.float32(params['angle'])
                        angle_idx = len(angle_params_array)
                        angle_params_array.append(angle)
                        angle_params_map[pid] = (k_idx, angle_idx)

                    angle_params_idxs.extend(angle_params_map[pid])
                    angle_atom_idxs.extend(atom_indices)

                angle_nrg = custom_ops.HarmonicAngleGPU_float(
                    angle_params_array,
                    list(range(start_params, start_params+len(angle_params_array))),
                    angle_params_idxs,
                    angle_atom_idxs
                )

                nrgs.append(angle_nrg)

                start_params += len(angle_params_array)
            elif force == 'PeriodicTorsionGenerator':

                torsion_params_map = {}
                torsion_params_array = []
                torsion_params_idxs = []
                torsion_atom_idxs = []

                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    # if pid not in torsion_params_map:
                    params = ff.getParameter(paramID=pid)
                    all_idxs = []
                    for order in range(1, 10):

                        k_str = "k"+str(order)
                        idivf_str = "idivf"+str(order)
                        phase_str = "phase"+str(order)
                        period_str = "periodicity"+str(order)

                        if k_str in params:

                            if pid not in torsion_params_map:

                                k = np.float32(params[k_str])/np.float32(params[idivf_str])
                                phase = np.float32(params[phase_str])
                                period = np.float32(params[period_str])

                                k_idx = len(torsion_params_array)
                                torsion_params_array.append(k)

                                phase_idx = len(torsion_params_array)
                                torsion_params_array.append(phase)

                                period_idx = len(torsion_params_array)
                                torsion_params_array.append(period)

                                torsion_params_map[pid] = (k_idx, phase_idx, period_idx)

                            torsion_params_idxs.extend(torsion_params_map[pid])
                            torsion_atom_idxs.extend(atom_indices)
                        else:
                            break

                torsion_nrg = custom_ops.PeriodicTorsionGPU_float(
                    torsion_params_array,
                    list(range(start_params, start_params+len(torsion_params_array))),
                    torsion_params_idxs,
                    torsion_atom_idxs
                )

                nrgs.append(torsion_nrg)

                start_params += len(torsion_params_array)

            elif force == 'NonbondedGenerator':
                print("\n%s:" % force)

                nbg = None
                for f in ff.getGenerators():
                    if isinstance(f, forcefield.NonbondedGenerator):
                        nbg = f

                assert nbg is not None

                lj14scale = nbg.lj14scale
                lj_params_map = {}
                lj_params_array = []
                lj_params_idxs = []

                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    if pid not in lj_params_map:
                        params = ff.getParameter(paramID=pid)
                        sigma = convert_rmin_half_to_sigma(np.float32(params['rmin_half']))
                        eps = np.float32(params['epsilon'])
                        sig_idx = len(lj_params_array)
                        lj_params_array.append(sigma)

                        eps_idx = len(lj_params_array)
                        lj_params_array.append(eps)

                        lj_params_map[pid] = (sig_idx, eps_idx)

                    lj_params_idxs.extend((lj_params_map[pid][0], lj_params_map[pid][1]))

                lj_scale_matrix = generate_scale_matrix(np.array(bond_atom_idxs).reshape(-1, 2),  lj14scale, N)

                lj_nrg = custom_ops.LennardJonesGPU_float(
                    lj_params_array,
                    list(range(start_params, start_params+len(lj_params_array))),
                    lj_params_idxs,
                    lj_scale_matrix.reshape(-1)
                )

                nrgs.append(lj_nrg)

                start_params += len(lj_params_array)

    return nrgs, start_params