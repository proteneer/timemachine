import math

from jax.config import config
from rdkit import Chem

config.update("jax_enable_x64", True)

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from timemachine.fe import topology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff.handlers import openmm_deserializer


@dataclass(eq=False)
class RABFEResult:
    mol_name: str
    dG_complex_conversion: float
    dG_complex_conversion_error: float
    dG_complex_decouple: float
    dG_complex_decouple_error: float
    dG_solvent_conversion: float
    dG_solvent_conversion_error: float
    dG_solvent_decouple: float
    dG_solvent_decouple_error: float

    def log(self):
        """print stage summary"""
        print(
            "stage summary for mol:",
            self.mol_name,
            "dG_complex_conversion (K complex)",
            self.dG_complex_conversion,
            "dG_complex_conversion_err",
            self.dG_complex_conversion_error,
            "dG_complex_decouple (E0 + A0 + A1 + E1)",
            self.dG_complex_decouple,
            "dG_complex_decouple_err",
            self.dG_complex_decouple_error,
            "dG_solvent_conversion (K solvent)",
            self.dG_solvent_conversion,
            "dG_solvent_conversion_err",
            self.dG_solvent_conversion_error,
            "dG_solvent_decouple (D)",
            self.dG_solvent_decouple,
            "dG_solvent_decouple_err",
            self.dG_solvent_decouple_error,
        )

    def __eq__(self, other: "RABFEResult") -> bool:
        if not isinstance(other, RABFEResult):
            return NotImplemented
        equal = True
        for field in fields(self):
            self_val = getattr(self, field.name)
            other_val = getattr(other, field.name)
            # Python doesn't consider nan == nan to be true
            if field.type is float and math.isnan(self_val) and math.isnan(other_val):
                continue
            equal &= self_val == other_val
        return equal

    @classmethod
    def _convert_field_to_sdf_field(cls, field_name: str) -> str:
        if field_name == "mol_name":
            cleaned_name = "_Name"
        else:
            cleaned_name = field_name.replace("_", " ")
        return cleaned_name

    @classmethod
    def from_mol(cls, mol: Chem.Mol):
        field_names = fields(cls)

        kwargs = {}
        for field in field_names:
            field_name = cls._convert_field_to_sdf_field(field.name)
            val = mol.GetProp(field_name)
            val = field.type(val)
            kwargs[field.name] = val
        return RABFEResult(**kwargs)

    def apply_to_mol(self, mol: Chem.Mol):
        results_dict = asdict(self)
        results_dict.update(
            {
                "dG_bind": self.dG_bind,
                "dG_bind_err": self.dG_bind_err,
            }
        )
        for field, val in results_dict.items():
            field_name = self._convert_field_to_sdf_field(field)
            mol.SetProp(field_name, str(val))

    @property
    def dG_complex(self):
        """effective free energy of removing from complex"""
        return self.dG_complex_conversion + self.dG_complex_decouple

    @property
    def dG_solvent(self):
        """effective free energy of removing from solvent"""
        return self.dG_solvent_conversion + self.dG_solvent_decouple

    @property
    def dG_bind(self):
        """the final value we seek is the free energy of moving
        from the solvent into the complex"""
        return self.dG_solvent - self.dG_complex

    @property
    def dG_bind_err(self):
        errors = np.asarray(
            [
                self.dG_complex_conversion_error,
                self.dG_complex_decouple_error,
                self.dG_solvent_conversion_error,
                self.dG_solvent_decouple_error,
            ]
        )
        return np.sqrt(np.sum(errors ** 2))


class UnsupportedTopology(Exception):
    pass


class BaseFreeEnergy:
    @staticmethod
    def _get_system_params_and_potentials(ff_params, topology):

        ff_tuples = [
            [topology.parameterize_harmonic_bond, (ff_params[0],)],
            [topology.parameterize_harmonic_angle, (ff_params[1],)],
            [topology.parameterize_periodic_torsion, (ff_params[2], ff_params[3])],
            [topology.parameterize_nonbonded, (ff_params[4], ff_params[5])],
        ]

        final_params = []
        final_potentials = []

        for fn, params in ff_tuples:
            combined_params, combined_potential = fn(*params)
            final_potentials.append(combined_potential)
            final_params.append(combined_params)

        return final_params, final_potentials


# this class is serializable.
class AbsoluteFreeEnergy(BaseFreeEnergy):
    def __init__(self, mol, top):
        """
        Compute the absolute free energy of a molecule via 4D decoupling.

        Parameters
        ----------
        mol: rdkit mol
            Ligand to be decoupled

        top: Topology
            topology.Topology to use

        """
        self.mol = mol
        self.top = top

    def prepare_host_edge(self, ff_params, host_system):
        """
        Prepares the host-edge system

        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params

        host_system: openmm.System
            openmm System object to be deserialized

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = [a.GetMass() for a in self.mol.GetAtoms()]
        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)

        combined_masses = np.concatenate([host_masses, ligand_masses])

        return final_potentials, final_params, combined_masses


# this class is serializable.
class RelativeFreeEnergy(BaseFreeEnergy):
    def __init__(self, dual_topology: topology.DualTopology, label=None, complex_path=None):
        self.top = dual_topology
        self.label = label
        self.complex_path = complex_path

    @property
    def mol_a(self):
        return self.top.mol_a

    @property
    def mol_b(self):
        return self.top.mol_b

    @property
    def ff(self):
        return self.top.ff

    def prepare_host_edge(self, ff_params, host_system):
        """
        Prepares the host-edge system
        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params
        host_system: openmm.System
            openmm System object to be deserialized

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """

        ligand_masses_a = [a.GetMass() for a in self.mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in self.mol_b.GetAtoms()]

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)

        combined_masses = np.concatenate([host_masses, ligand_masses_a, ligand_masses_b])

        return final_potentials, final_params, combined_masses


def validate_lambda_schedule(lambda_schedule, num_windows):
    """Must go monotonically from 0 to 1 in num_windows steps"""
    assert lambda_schedule[0] == 0.0
    assert lambda_schedule[-1] == 1.0
    assert len(lambda_schedule) == num_windows
    assert ((lambda_schedule[1:] - lambda_schedule[:-1]) > 0).all()


def interpolate_pre_optimized_protocol(pre_optimized_protocol, num_windows):
    xp = np.linspace(0, 1, len(pre_optimized_protocol))
    x_interp = np.linspace(0, 1, num_windows)
    lambda_schedule = np.interp(x_interp, xp, pre_optimized_protocol)

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_conversion_lambda_schedule(num_windows):
    lambda_schedule = np.linspace(0, 1, num_windows)
    validate_lambda_schedule(lambda_schedule, num_windows)
    return lambda_schedule


def construct_absolute_lambda_schedule_complex(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded_cutoff = 1.2
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.20 * num_windows)
    B = int(0.50 * num_windows)
    C = num_windows - A - B

    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.1, A, endpoint=False),
            np.linspace(0.1, 0.3, B, endpoint=False),
            np.linspace(0.3, 1.0, C, endpoint=True),
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_absolute_lambda_schedule_solvent(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * schedule assumes nonbonded_cutoff = 1.2
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.20 * num_windows)
    B = int(0.66 * num_windows)
    D = 1  # need only one window from 0.6 to 1.0
    C = num_windows - A - B - D

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.08, A, endpoint=False),
            np.linspace(0.08, 0.27, B, endpoint=False),
            np.linspace(0.27, 0.50, C, endpoint=True),
            [1.0],
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_pre_optimized_absolute_lambda_schedule_solvent(num_windows, nonbonded_cutoff=1.2):
    """Linearly interpolate a lambda schedule pre-optimized for solvent decoupling

    Notes
    -----
    * Generated by post-processing ~half a dozen solvent decoupling calculations
        (see context in description of PR #538)
    * Assumes nonbonded cutoff = 1.2 nm
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """

    assert nonbonded_cutoff == 1.2

    # fmt: off
    solvent_decoupling_protocol = np.array(
        [0., 0.02154097, 0.0305478, 0.03747918, 0.0432925, 0.04841349, 0.05303288, 0.05729336, 0.06128111, 0.0650162,
         0.06854392, 0.07186945, 0.07505386, 0.07809426, 0.08097656, 0.08378378, 0.08652228, 0.08910844, 0.09170097,
         0.09415532, 0.0965975, 0.09894146, 0.10125901, 0.10349315, 0.1057036, 0.10782406, 0.10995297, 0.11196338,
         0.11404105, 0.11597311, 0.11799029, 0.11989214, 0.12179616, 0.12367442, 0.12544245, 0.12730977, 0.12904358,
         0.13080329, 0.13255268, 0.13418286, 0.13594787, 0.13760607, 0.13920917, 0.14090233, 0.14247115, 0.14403571,
         0.14563762, 0.14712597, 0.14863463, 0.1501709, 0.1516045, 0.15306237, 0.15457974, 0.15599668, 0.15739867,
         0.1588833, 0.1602667, 0.16158698, 0.16306219, 0.16443643, 0.16571203, 0.1671053, 0.16844875, 0.16969885,
         0.17095515, 0.17229892, 0.17355947, 0.17474395, 0.17606238, 0.17735235, 0.1785562, 0.1797194, 0.18102615,
         0.18224503, 0.18338315, 0.18454735, 0.18579297, 0.18695968, 0.18805265, 0.18920557, 0.1904094, 0.1915372,
         0.1925929, 0.19370481, 0.19486737, 0.19595772, 0.19698288, 0.19803636, 0.1991899, 0.20028, 0.20131035,
         0.20232168, 0.20348772, 0.20458663, 0.2056212, 0.20659485, 0.20774405, 0.20884764, 0.20989276, 0.2108857,
         0.2120116, 0.21316817, 0.21427184, 0.21532528, 0.21650709, 0.21773745, 0.21890783, 0.22002229, 0.22133134,
         0.2226356, 0.22387771, 0.22515419, 0.22662608, 0.22803088, 0.22940172, 0.23108277, 0.2327005, 0.23438922,
         0.23634133, 0.23822652, 0.2405842, 0.24292293, 0.24588996, 0.24922462, 0.25322387, 0.25836924, 0.26533154,
         0.27964026, 0.29688698, 0.31934273, 0.34495637, 0.37706286, 0.4246625, 0.5712542, 1.]
    )
    # fmt: on

    lambda_schedule = interpolate_pre_optimized_protocol(solvent_decoupling_protocol, num_windows)

    return lambda_schedule


def construct_relative_lambda_schedule(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded cutoff = 1.2 nm
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """

    A = int(0.15 * num_windows)
    B = int(0.60 * num_windows)
    C = num_windows - A - B

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.00, 0.08, A, endpoint=False),
            np.linspace(0.08, 0.27, B, endpoint=False),
            np.linspace(0.27, 1.00, C, endpoint=True),
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def setup_relative_restraints_by_distance(
    mol_a: Chem.Mol, mol_b: Chem.Mol, cutoff: float = 0.1, terminal: bool = False
):
    """
    Setup restraints between atoms in two molecules using
    a cutoff distance between atoms

    Parameters
    ----------
    mol_a: Chem.Mol
        First molecule

    mol_b: Chem.Mol
        Second molecule

    cutoff: float=0.1
        Distance between atoms to consider as a match

    terminal: bool=false
        Map terminal atoms

    Returns
    -------
    np.array (N, 2)
        Atom mapping between atoms in mol_a to atoms in mol_b.
    """

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)
    core_idxs_a = []
    core_idxs_b = []

    for idx, a in enumerate(mol_a.GetAtoms()):
        if not terminal and a.GetDegree() == 1:
            continue
        for b_idx, b in enumerate(mol_b.GetAtoms()):
            if not terminal and b.GetDegree() == 1:
                continue
            if np.linalg.norm(ligand_coords_a[idx] - ligand_coords_b[b_idx]) < cutoff:
                core_idxs_a.append(idx)
                core_idxs_b.append(b_idx)
    assert len(core_idxs_a) == len(core_idxs_b), "Core sizes were inconsistent"

    rij = cdist(ligand_coords_a[core_idxs_a], ligand_coords_b[core_idxs_b])

    row_idxs, col_idxs = linear_sum_assignment(rij)

    core_idxs = []

    for core_a, core_b in zip(row_idxs, col_idxs):
        core_idxs.append((core_idxs_a[core_a], core_idxs_b[core_b]))

    core_idxs = np.array(core_idxs, dtype=np.int32)

    return core_idxs


def setup_relative_restraints_using_smarts(mol_a, mol_b, smarts):
    """
    Setup restraints between atoms in two molecules using
    a pre-defined SMARTS pattern.

    Parameters
    ----------
    mol_a: Chem.Mol
        First molecule

    mol_b: Chem.Mol
        Second molecule

    smarts: string
        Smarts pattern defining the common core.

    Returns
    -------
    np.array (N, 2)
        Atom mapping between atoms in mol_a to atoms in mol_b.

    """

    # check to ensure the core is connected
    # technically allow for this but we need to do more validation before
    # we can be fully comfortable
    assert "." not in smarts

    core = Chem.MolFromSmarts(smarts)

    # we want *all* possible combinations.
    limit = 1000
    all_core_idxs_a = np.array(mol_a.GetSubstructMatches(core, uniquify=False, maxMatches=limit))
    all_core_idxs_b = np.array(mol_b.GetSubstructMatches(core, uniquify=False, maxMatches=limit))

    assert len(all_core_idxs_a) < limit
    assert len(all_core_idxs_b) < limit

    best_rmsd = np.inf
    best_core_idxs_a = None
    best_core_idxs_b = None

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    # setup relative orientational restraints
    # rough sketch of algorithm:
    # find core atoms in mol_a
    # find core atoms in mol_b
    # for all matches in mol_a
    #    for all matches in mol_b
    #       use the hungarian algorithm to assign matching
    #       if sum is smaller than best, then store.

    for core_idxs_a in all_core_idxs_a:
        for core_idxs_b in all_core_idxs_b:

            ri = np.expand_dims(ligand_coords_a[core_idxs_a], 1)
            rj = np.expand_dims(ligand_coords_b[core_idxs_b], 0)
            rij = np.sqrt(np.sum(np.power(ri - rj, 2), axis=-1))

            row_idxs, col_idxs = linear_sum_assignment(rij)

            rmsd = np.linalg.norm(ligand_coords_a[core_idxs_a[row_idxs]] - ligand_coords_b[core_idxs_b[col_idxs]])

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_core_idxs_a = core_idxs_a
                best_core_idxs_b = core_idxs_b

    core_idxs = np.stack([best_core_idxs_a, best_core_idxs_b], axis=1).astype(np.int32)
    print("core_idxs", core_idxs, "rmsd", best_rmsd)

    return core_idxs
