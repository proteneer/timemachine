import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import special_ortho_group

from timemachine import constants


def convert_uIC50_to_kJ_per_mole(amount_in_uM: float, experiment_temp: float = constants.DEFAULT_TEMP) -> float:
    """Convert an IC50 measurement in uM concentrations to kJ/mol.

    Parameters
    ----------

    amount_in_uM: float
        Micro molar IC50

    experiment_temp: float
        Experiment temperature in Kelvin. Defaults to constants.DEFAULT_TEMP

    Returns
    -------
    float
        Binding potency in kJ/mol.

    """
    RT = (constants.BOLTZ * experiment_temp) / constants.KCAL_TO_KJ
    return RT * np.log(amount_in_uM * 1e-6) * constants.KCAL_TO_KJ


def convert_uM_to_kJ_per_mole(amount_in_uM: float, experiment_temp: float = constants.DEFAULT_TEMP) -> float:
    """
    Convert a potency measurement in uM concentrations to kJ/mol.

    Parameters
    ----------
    amount_in_uM: float
        Binding potency in uM concentration.

    experiment_temp: float
        Experiment temperature in Kelvin. Defaults to constants.DEFAULT_TEMP

    Returns
    -------
    float
        Binding potency in kJ/mol.

    """
    return convert_uIC50_to_kJ_per_mole(amount_in_uM, experiment_temp=experiment_temp)


# TODO: add a module for atom-mapping, with RDKit MCS based and other approaches

# TODO: add a visualization module?
# TODO: compare with perses atom map visualizations?


def draw_mol(mol, highlightAtoms, highlightColors):
    """from YTZ, Feb 1, 2021"""
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
    drawer.DrawMolecule(mol, highlightAtoms=highlightAtoms, highlightAtomColors=highlightColors)
    drawer.FinishDrawing()

    # TODO: return or save image, for inclusion in a PDF report or similar

    # To display in a notebook:
    # svg = drawer.GetDrawingText().replace('svg:', '')
    # display(SVG(svg))


def draw_mol_idx(mol, highlight: Optional[list[int]] = None, scale_factor=None):
    """
    Draw mol with atom indices labeled.

    Pararmeters
    -----------
    highlight: List of int or None
        If specified, highlight the given atom idxs.
    """
    mol2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol2d)
    if scale_factor:
        AllChem.NormalizeDepiction(mol2d, scaleFactor=scale_factor)
    for atom in mol2d.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    return Draw.MolsToGridImage(
        [mol2d],
        molsPerRow=1,
        highlightAtomLists=[highlight] if highlight is not None else None,
        subImgSize=(500, 500),
        legends=[get_mol_name(mol2d)],
        useSVG=True,
    )


def get_atom_map_colors(core, seed=2022):
    rng = np.random.default_rng(seed)

    atom_colors_a = {}
    atom_colors_b = {}
    # TODO: replace random colors with colormap?
    for (a_idx, b_idx), rgb in zip(core, rng.random((len(core), 3))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())

    return atom_colors_a, atom_colors_b


def plot_atom_mapping(mol_a, mol_b, core, seed=2022):
    """TODO: move this into a SingleTopology.visualize() or SingleTopology.debug() method?"""

    atom_colors_a, atom_colors_b = get_atom_map_colors(core, seed)

    draw_mol(mol_a, core[:, 0].tolist(), atom_colors_a)
    draw_mol(mol_b, core[:, 1].tolist(), atom_colors_b)


def recenter_mol(mol):
    mol_copy = Chem.Mol(mol)
    conf = mol.GetConformer(0).GetPositions()
    center_conf = conf - np.mean(conf, axis=0)
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, pos in enumerate(np.asarray(center_conf)):
        new_conf.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(new_conf)
    return mol_copy


def score_2d(conf, norm=2):
    # get the goodness of a 2D depiction
    # low_score = good, high_score = bad

    score = 0
    for idx, (x0, y0, _) in enumerate(conf):
        for x1, y1, _ in conf[idx + 1 :]:
            score += 1 / ((x0 - x1) ** norm + (y0 - y1) ** norm)

    return score / len(conf)


def generate_good_rotations(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    num_rotations: int = 3,
    max_rotations: int = 1000,
    seed: int = 1234,
) -> NDArray:
    assert num_rotations < max_rotations

    # generate some good rotations so that the viewing angle is pleasant, (so clashes are minimized):
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    unif_so3 = special_ortho_group(dim=3, seed=seed)

    scores = []
    rotations = []
    for _ in range(max_rotations):
        r = unif_so3.rvs()
        score_a = score_2d(conf_a @ r.T)
        score_b = score_2d(conf_b @ r.T)
        # take the bigger of the two scores
        scores.append(max(score_a, score_b))
        rotations.append(r)

    perm = np.argsort(scores, kind="stable")
    return np.array(rotations)[perm][:num_rotations]


def rotate_mol(mol, rotation_matrix):
    mol = recenter_mol(mol)
    conf = mol.GetConformer(0).GetPositions()

    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, pos in enumerate(np.asarray(conf)):
        rot_pos = rotation_matrix @ pos
        new_conf.SetAtomPosition(idx, (float(rot_pos[0]), float(rot_pos[1]), float(rot_pos[2])))

    mol_copy = Chem.Mol(mol)
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(new_conf)
    return mol_copy


def generate_bond_idxs_and_colors(mol_a, mol_b, core):
    bond_core_on = (144 / 255, 238 / 255, 144 / 255)  # green
    bond_core_off = (238 / 255, 144 / 255, 144 / 255)  # red
    bond_idxs_a = []
    bond_colors_a = dict()
    core_a_to_b = dict(core)
    for bond in mol_a.GetBonds():
        src_a, dst_a = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if src_a in core_a_to_b and dst_a in core_a_to_b:
            bond_idxs_a.append(bond.GetIdx())
            if mol_b.GetBondBetweenAtoms(int(core_a_to_b[src_a]), int(core_a_to_b[dst_a])):
                bond_colors_a[int(bond.GetIdx())] = bond_core_on
            else:
                bond_colors_a[int(bond.GetIdx())] = bond_core_off

    return bond_idxs_a, bond_colors_a


def plot_atom_mapping_grid(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    num_rotations: int = 5,
    seed: int = 1234,
):
    mol_a_3d = recenter_mol(mol_a)
    mol_b_3d = recenter_mol(mol_b)

    extra_rotations = generate_good_rotations(mol_a, mol_b, num_rotations, seed=seed)

    extra_mols = []

    atom_colors_a = {}
    atom_colors_b = {}
    for (a_idx, b_idx), rgb in zip(core, np.random.random((len(core), 3))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())  # type: ignore[arg-type]
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())  # type: ignore[arg-type]

    # highlight bond idxs and colors
    bond_idxs_a, bond_colors_a = generate_bond_idxs_and_colors(mol_a, mol_b, core)
    bond_idxs_b, bond_colors_b = generate_bond_idxs_and_colors(mol_b, mol_a, core[:, [1, 0]])
    # highlight atom idxs and colors
    hals = [core[:, 0].tolist(), core[:, 1].tolist()]
    hacs = [atom_colors_a, atom_colors_b]

    hbls = [bond_idxs_a, bond_idxs_b]
    hbcs = [bond_colors_a, bond_colors_b]

    for rot in extra_rotations:
        extra_mols.append(rotate_mol(mol_a_3d, rot))
        extra_mols.append(rotate_mol(mol_b_3d, rot))
        hals.append(core[:, 0].tolist())
        hals.append(core[:, 1].tolist())
        hacs.append(atom_colors_a)
        hacs.append(atom_colors_b)
        hbls.append(bond_idxs_a)
        hbls.append(bond_idxs_b)
        hbcs.append(bond_colors_a)
        hbcs.append(bond_colors_b)

    all_mols = [mol_a_3d, mol_b_3d, *extra_mols]
    num_mols = len(all_mols)

    legends: list[str] = []
    while len(legends) < num_mols:
        legends.append(get_mol_name(mol_a) + " (3D)")
        legends.append(get_mol_name(mol_b) + " (3D)")

    return Draw.MolsToGridImage(
        all_mols,
        molsPerRow=num_mols,
        highlightAtomLists=hals,
        highlightAtomColors=hacs,
        highlightBondLists=hbls,
        highlightBondColors=hbcs,
        subImgSize=(25 * num_mols, 300),
        legends=legends,
        useSVG=True,
    )


type _Core = Sequence[Sequence[int]] | NDArray


def view_atom_mapping_3d(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    cores: Sequence[_Core] | NDArray = (),
    colors: Sequence[str] = (
        # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ),
    show_atom_idx_labels: bool = False,
    width: int = 800,
    row_height: int = 200,
    seed: int = 0,
):
    """Produce a 3D rotatable view of a pair of molecules using py3Dmol. If `cores` is nonempty, display additional rows
    where the atoms are colored according to the atom mapping.

    Parameters
    ----------
    mol_a, mol_b : rdkit mols
        Input mols

    cores : list of list of list of int or ndarray, optional
        Atom mappings. If nonempty, display additional rows with atoms color-coded according to the corresponding
        mapping

    colors : list of str, optional
        Colors to use for highlighting atoms by mapping

    show_atom_idx_labels : bool, optional
        Whether to display atom indices

    width : int, optional
        Width of the view

    row_height : int, optional
        Height of each row of the view

    seed : int, optional
        RNG seed used to generate color ordering

    Returns
    -------
    py3Dmol.view
    """

    try:
        import py3Dmol
    except ImportError as e:
        raise RuntimeError("requires py3Dmol to be installed") from e

    for core in cores:
        assert np.asarray(core).ndim == 2, "expect a list of cores"

    def make_style(props):
        return {"stick": props}

    def atom_style(color):
        return make_style({"color": color})

    dummy_style = atom_style("white")

    num_rows = 1 + len(cores)
    height = num_rows * row_height
    view = py3Dmol.view(viewergrid=(num_rows, 2), width=width, height=height)

    def add_mol(mol, viewer):
        view.addModel(Chem.MolToMolBlock(mol), "mol", viewer=viewer)

    add_mol(mol_a, (0, 0))
    add_mol(mol_b, (0, 1))

    view.setStyle(make_style({}))

    # additional rows, colored according to corresponding mappings
    for row, core in enumerate(cores, 1):
        add_mol(mol_a, (row, 0))
        add_mol(mol_b, (row, 1))

        view.setStyle(dummy_style, viewer=(row, 0))
        view.setStyle(dummy_style, viewer=(row, 1))

        rng = np.random.default_rng(seed)
        colors_ = (
            rng.permutation(colors)
            if len(core) <= len(colors)
            # if more atoms than colors, need to reuse some colors
            else rng.choice(colors, len(core), replace=True)
        )

        if isinstance(core, np.ndarray):
            core = core.tolist()

        for (ia, ib), color in zip(core, colors_):
            view.setStyle({"serial": ia}, atom_style(color), viewer=(row, 0))
            view.setStyle({"serial": ib}, atom_style(color), viewer=(row, 1))

    view.zoomTo()

    if show_atom_idx_labels:
        view.addPropertyLabels("serial", "", {"alignment": "center", "fontSize": 10})

    return view


def view_rest_region_3d(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    rest_region_atom_idxs_a: Sequence[int],
    rest_region_atom_idxs_b: Sequence[int],
    show_atom_idx_labels: bool = False,
):
    try:
        import py3Dmol
    except ImportError as e:
        raise RuntimeError("requires py3Dmol to be installed") from e

    def make_style(props):
        return {"stick": props}

    def atom_style(color):
        return make_style({"color": color})

    view = py3Dmol.view(viewergrid=(2, 2))

    def add_mol(mol, viewer):
        view.addModel(Chem.MolToMolBlock(mol), "mol", viewer=viewer)

    add_mol(mol_a, (0, 0))
    add_mol(mol_b, (0, 1))
    view.setStyle(make_style({}))

    add_mol(mol_a, (1, 0))
    view.setStyle(atom_style("white"), viewer=(1, 0))
    for idx in rest_region_atom_idxs_a:
        view.setStyle({"serial": idx}, {"stick": {"color": "red"}}, viewer=(1, 0))

    add_mol(mol_b, (1, 1))
    view.setStyle(atom_style("white"), viewer=(1, 1))
    for idx in rest_region_atom_idxs_b:
        view.setStyle({"serial": idx}, atom_style("red"), viewer=(1, 1))

    view.zoomTo()

    if show_atom_idx_labels:
        view.addPropertyLabels("serial", "", {"alignment": "center", "fontSize": 10})

    return view


def get_romol_bonds(mol):
    """
    Return bond idxs given a mol. These are not canonicalized.
    """
    bond_list = []
    for bond in mol.GetBonds():
        bond_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bond_list


def get_romol_conf(mol, conf_id: int = 0) -> NDArray:
    """Coordinates of the specified conformer, in nanometers"""
    conformer = mol.GetConformer(conf_id)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf / 10.0  # from angstroms to nm


def set_romol_conf(mol, new_coords: NDArray, conf_id: int = 0):
    """Sets coordinates of the specified conformer. Expects coords in nanometers and converts to angstroms"""
    assert new_coords.shape[0] == mol.GetNumAtoms()
    # convert from nm to angstroms
    angstrom_coords = new_coords * 10.0
    angstrom_coords = angstrom_coords.astype(np.float64)  # Must be float64
    conf = mol.GetConformer(conf_id)
    for i, pos in enumerate(angstrom_coords):
        conf.SetAtomPosition(i, pos)


def get_mol_masses(mol) -> NDArray:
    """Return the masses for the given mol"""
    return np.array([a.GetMass() for a in mol.GetAtoms()])


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


def set_mol_name(mol, name: str):
    """Set an RDKit mol's name"""
    mol.SetProp("_Name", name)


def sanitize_energies(full_us, lamb_idx, cutoff=10000):
    """
    Given a matrix with F rows and K columns,
    we sanitize entries that differ by more than cutoff.

    That is, given full_us:
    [
        [15000.0, -5081923.0, 1598, 1.5, -23.0],
        [-423581.0, np.nan, -238, 13.5,  23.0]
    ]
    And lamb_idx 3 and cutoff of 10000,
    full_us is sanitized to:

    [
        [inf, inf, 1598, 1.5, -23.0],
        [inf, inf, -238, 13.5,  23.0]
    ]

    Parameters
    ----------
    full_us: NDArray of shape (F, K)
        Matrix of full energies

    lamb_idx: int
        Which of the K windows to serve as the reference energy

    cutoff: float
        Used to determine the threshold for a "good" energy

    Returns
    -------
    np.array of shape (F,K)
        Sanitized energies

    """
    ref_us = np.expand_dims(full_us[:, lamb_idx], axis=1)
    abs_us = np.abs(full_us - ref_us)
    return np.where(abs_us < cutoff, full_us, np.inf)


def read_sdf(fname: str | Path, removeHs: bool = False) -> list[Chem.Mol]:
    """Read list of mols from an SDF

    Parameters
    ----------
    fname: str or Path
        Path to the SDF file

    removeHs: bool
        Remove hydrogens from mol, defaults to False

    Returns
    -------
        List of mols
            By default maintains hydrogens, unless removeHs=True
    """
    supplier = Chem.SDMolSupplier(str(fname), removeHs=removeHs)
    mols = [mol for mol in supplier]
    return mols


def read_sdf_mols_by_name(fname: str | Path, removeHs: bool = False) -> dict[str, Chem.Mol]:
    mols = read_sdf(fname, removeHs)
    mols_by_name = {get_mol_name(mol): mol for mol in mols}
    return mols_by_name


def extract_delta_Us_from_U_knk(U_knk):
    """
    Generate delta_Us from the U_knk matrix for use with BAR.

    Parameters
    ----------
    U_knk: NDArray of shape (K, N, K)
        Energies matrix, K simulations ran with N frames with
        energies evaluated at K states

    Returns
    -------
    np.array of shape (K-1, 2, N)
        Returns the delta_Us of the fwd and rev processes

    """

    assert U_knk.shape[0] == U_knk.shape[-1]

    K = U_knk.shape[0]

    def delta_U(from_idx, to_idx):
        """
        Computes [U(x, to_idx) - U(x, from_idx) for x in xs]
        where xs are simulated at from_idx
        """
        current = U_knk[from_idx]
        current_energies = current[:, from_idx]
        perturbed_energies = current[:, to_idx]
        return perturbed_energies - current_energies

    delta_Us = []

    for lambda_idx in range(K - 1):
        # lambda_us have shape (F, K)
        fwd_delta_U = delta_U(lambda_idx, lambda_idx + 1)
        rev_delta_U = delta_U(lambda_idx + 1, lambda_idx)
        delta_Us.append((fwd_delta_U, rev_delta_U))

    return np.array(delta_Us)


def bytes_to_id(data: bytes) -> int:
    # Convert the given data into a 64-bit int
    MAX_INT = 2**64 - 1
    return int(hashlib.sha256(data).hexdigest(), 16) % MAX_INT
