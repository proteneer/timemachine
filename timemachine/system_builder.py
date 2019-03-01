from simtk import unit

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


def _ast_eval(node):
    """
    Performs an algebraic syntax tree evaluation of a unit.
    """

    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
        ast.USub: op.neg}

    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](_ast_eval(node.operand))
    elif isinstance(node, ast.Name):
        # see if this is a simtk unit
        b = getattr(unit, node.id)
        return b
    else:
        raise TypeError(node)

def _extractQuantity(node, parent, name, unit_name=None):
    """
    Form a (potentially unit-bearing) quantity from the specified attribute name.

    node : xml.etree.ElementTree.Element
       Node of etree corresponding to force type entry.
    parent : xml.etree.ElementTree.Element
       Node of etree corresponding to parent Force.
    name : str
       Name of parameter to extract from attributes.
    unit_name : str, optional, default=None
       If specified, use this attribute name of 'parent' to look up units

    """
    # print(node, parent, name, unit_name)
    # Check for expected attributes
    if name not in node.attrib:
        if 'sourceline' in node.attrib:
            raise Exception("Line %d : Expected XML attribute '%s' not found" % (node.attrib['sourceline'], name))
        else:
            raise Exception("Expected XML attribute '%s' not found" % (name))

    # Most attributes will be converted to floats, but some are strings
    string_names = ['parent_id', 'id']
    # Handle case where this is a normal quantity
    if name not in string_names:
        quantity = float(node.attrib[name])
    # Handle label or string
    else:
        quantity = node.attrib[name]
        return quantity

    if unit_name is None:
        unit_name = name + '_unit'

    if unit_name in parent.attrib:
        a = node.attrib[name]
        b = parent.attrib[unit_name]
        parsed_units = _ast_eval(ast.parse(b, mode='eval').body)
        result = float(a) * parsed_units
        quantity = result.value_in_unit_system(unit.md_unit_system)

    return quantity

def construct_energies(ff, mol, am1_charges=True):
    """
    Construct energies given a forcefield and a molecule.

    Parameters
    ----------
    ff: openforcefield.ForceField
        Pre-loaded forcefield.

    mol: oechem.OEMol
        OpenEye Mol object

    charges: bool
        Whether or not we apply charges 

    Returns
    -------
    list of energy object

    """
    labels = ff.labelMolecules([mol], verbose=False)
    N = mol.NumAtoms()
    start_params = 0
    nrgs = []
    offsets = []

    gens = ff.getGenerators()

    # very poorly thought out loops
    def get_bonded_term(pid):
        for b in gens[0]._bondtypes:
            if b.pid == pid:
                return b.k, b.length
        assert 0

    def get_angle_term(pid):
        for b in gens[1]._angletypes:
            if b.pid == pid:
                return b.k, b.angle
        assert 0

    def get_torsion_term(pid):
        # print("PID", pid) # improper torsions
        for b in gens[2]._propertorsiontypes:
            if b.pid == pid:
                return b.k, b.phase, b.periodicity
        for b in gens[2]._impropertorsiontypes:
            if b.pid == pid:
                return b.k, b.phase, b.periodicity
        assert 0

    def get_nonbonded_term(pid):
        for b in gens[3]._ljtypes:
            if b.pid == pid:
                return b.sigma, b.epsilon, b.charge
        assert 0


    for mol_entry in range(len(labels)):

        for force in labels[mol_entry].keys():
            print("PARSING", force)
            if force == 'HarmonicBondGenerator':
                bond_params_map = {}
                bond_params_array = []
                bond_params_idxs = []
                bond_atom_idxs = []
                for (atom_indices, pid, smirks) in labels[mol_entry][force]:

                    if pid not in bond_params_map:
                        k, length = get_bonded_term(pid)
                        k_idx = len(bond_params_array)
                        bond_params_array.append(k)
                        length_idx = len(bond_params_array)
                        bond_params_array.append(length)
                        bond_params_map[pid] = (k_idx, length_idx)

                    bond_params_idxs.extend(bond_params_map[pid])
                    bond_atom_idxs.extend(atom_indices)

                # print(bond_params_array, list(range(start_params, start_params+len(bond_params_array))), bond_params_idxs, bond_atom_idxs)

                bond_nrg = (
                    # custom_ops.HarmonicBondGPU_double,
                    bond_params_array,
                    list(range(start_params, start_params+len(bond_params_array))),
                    bond_params_idxs,
                    bond_atom_idxs
                )

                nrgs.append(bond_nrg)


                offsets.append(start_params)
                start_params += len(bond_params_array)
            elif force == 'HarmonicAngleGenerator':
                # assert 0
                # continue
                angle_params_map = {}
                angle_params_array = []
                angle_params_idxs = []
                angle_atom_idxs = []

                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    if pid not in angle_params_map:
                        k, angle = get_angle_term(pid)
                        k_idx = len(angle_params_array)
                        angle_params_array.append(k)

                        angle_idx = len(angle_params_array)
                        angle_params_array.append(angle)
                        angle_params_map[pid] = (k_idx, angle_idx)

                    angle_params_idxs.extend(angle_params_map[pid])
                    angle_atom_idxs.extend(atom_indices)

                angle_nrg = (
                    # custom_ops.HarmonicAngleGPU_double,
                    angle_params_array,
                    list(range(start_params, start_params+len(angle_params_array))),
                    angle_params_idxs,
                    angle_atom_idxs
                )

                nrgs.append(angle_nrg)
                offsets.append(start_params)
                start_params += len(angle_params_array)
            elif force == 'PeriodicTorsionGenerator':
                torsion_params_map = {}
                torsion_params_array = []
                torsion_params_idxs = []
                torsion_atom_idxs = []

                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    # params = ff.getParameter(paramID=pid)
                    if pid not in torsion_params_map:
                        ks, phases, periods = get_torsion_term(pid)
                        all_terms = []
                        for k, phase, period in zip(ks, phases, periods):
                            k_idx = len(torsion_params_array)
                            torsion_params_array.append(k)

                            phase_idx = len(torsion_params_array)
                            torsion_params_array.append(phase)

                            period_idx = len(torsion_params_array)
                            torsion_params_array.append(period)

                            all_terms.append((k_idx, phase_idx, period_idx))

                        # print("inserting torsional parameter", pid)
                        torsion_params_map[pid] = all_terms

                    for k_idx, phase_idx, period_idx in torsion_params_map[pid]:
                        torsion_params_idxs.extend((k_idx, phase_idx, period_idx))
                        torsion_atom_idxs.extend(atom_indices)

                torsion_nrg = (
                    # custom_ops.PeriodicTorsionGPU_double,
                    torsion_params_array,
                    list(range(start_params, start_params+len(torsion_params_array))),
                    torsion_params_idxs,
                    torsion_atom_idxs
                )

                nrgs.append(torsion_nrg)
                offsets.append(start_params)
                start_params += len(torsion_params_array)                                                                                                                                                       

            elif force == 'NonbondedGenerator':
                # continue
                # print("\n%s:" % force)

                nbg = None
                for f in ff.getGenerators():
                    if isinstance(f, forcefield.NonbondedGenerator):
                        nbg = f

                assert nbg is not None

                es14scale = nbg.coulomb14scale
                lj14scale = nbg.lj14scale                      
                lj_params_map = {}
                lj_params_array = []
                lj_params_idxs = [None]*N

                global_charge_idxs = []

                charge_params_map = {}
                charge_params_array = []
                charge_params_idxs = [None]*N

                for (atom_indices, pid, smirks) in labels[mol_entry][force]:
                    if pid not in lj_params_map:                    
                        global_charge_idxs.append(int(pid[1:])-1)
                        sigma, eps, charge = get_nonbonded_term(pid)
                        sig_idx = len(lj_params_array)
                        lj_params_array.append(sigma)
                        eps_idx = len(lj_params_array)
                        lj_params_array.append(eps)
                        lj_params_map[pid] = (sig_idx, eps_idx)

                        # if not am1_charges:
                        charge_idx = len(charge_params_array)
                        charge_params_array.append(charge)
                        charge_params_map[pid] = charge_idx

                    lj_params_idxs[atom_indices[0]] = (lj_params_map[pid][0], lj_params_map[pid][1])
                    charge_params_idxs[atom_indices[0]] = charge_params_map[pid]

                lj_scale_matrix = generate_scale_matrix(np.array(bond_atom_idxs).reshape(-1, 2),  lj14scale, N)
                lj_nrg = (
                    # custom_ops.LennardJonesGPU_double,
                    lj_params_array,
                    list(range(start_params, start_params+len(lj_params_array))),
                    np.array(lj_params_idxs).reshape(-1),
                    lj_scale_matrix.reshape(-1)
                )

                nrgs.append(lj_nrg)
                offsets.append(start_params)
                start_params += len(lj_params_array)

                # generate charges using am1bcc
                if am1_charges:
                    print("Using am1 charges")
                    ff._assignPartialCharges(mol, "OECharges_AM1BCCSym")

                    es_scale_matrix = generate_scale_matrix(
                        np.array(bond_atom_idxs).reshape(-1, 2),
                        es14scale,
                        N
                    )

                    am1_charge_params = []
                    am1_charge_idxs = []

                    for atom_idx, atom in enumerate(mol.GetAtoms()):
                        am1_charge_params.append((atom.GetPartialCharge()*unit.elementary_charge).value_in_unit_system(unit.md_unit_system))
                        am1_charge_idxs.append(atom_idx)

                    print("True am1 charges:", am1_charge_params)
                    charge_nrg = (
                        # custom_ops.ElectrostaticsGPU_double,
                        am1_charge_params,
                        list(range(start_params, start_params+len(am1_charge_params))),
                        am1_charge_idxs,
                        es_scale_matrix.reshape(-1)
                    )

                    nrgs.append(charge_nrg)
                    offsets.append(start_params)
                    start_params += len(am1_charge_params)

                # use vanilla charges
                else:
                    # print("Using atom-typed charges", charge_params_array, charge_params_idxs)
                    es_scale_matrix = generate_scale_matrix(
                        np.array(bond_atom_idxs).reshape(-1, 2),
                        es14scale,
                        N
                    )
                    charge_nrg = (
                        # custom_ops.ElectrostaticsGPU_double,
                        charge_params_array,
                        list(range(start_params, start_params+len(charge_params_array))),
                        charge_params_idxs,
                        es_scale_matrix.reshape(-1)
                    )

                    nrgs.append(charge_nrg)
                    offsets.append(start_params)
                    start_params += len(charge_params_array)

    return nrgs, start_params, offsets, global_charge_idxs