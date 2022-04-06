import math

from jax.config import config
from rdkit import Chem

config.update("jax_enable_x64", True)

from dataclasses import asdict, dataclass, fields
from typing import Union

import numpy as np

from timemachine.fe import topology
from timemachine.fe.utils import get_mol_masses, get_romol_conf
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
        Prepares the host-guest system

        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params

        host_system: openmm.System
            openmm System object to be deserialized.

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)
        combined_masses = self._combine(ligand_masses, host_masses)
        return final_potentials, final_params, combined_masses

    def prepare_vacuum_edge(self, ff_params):
        """
        Prepares the vacuum system

        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)
        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, self.top)
        return final_potentials, final_params, ligand_masses

    def prepare_combined_coords(self, host_coords=None):
        """
        Returns the combined coordinates.

        Parameters
        ----------
        host_coords: np.array
            Nx3 array of atomic coordinates
            If None, return just the ligand coordinates.

        Returns
        -------
            combined_coordinates
        """
        ligand_coords = get_romol_conf(self.mol)
        return self._combine(ligand_coords, host_coords)

    def _combine(self, ligand_values, host_values=None):
        """
        Combine the values along the 0th axis.
        The host values will be first, if given.
        Then ligand values.

        Parameters
        ----------
        ligand_values: np.array
        host_values: Optional[np.array]

        Returns
        -------
            combined_values
        """
        if host_values is None:
            return ligand_values
        return np.concatenate([host_values, ligand_values])


# this class is serializable.
class RelativeFreeEnergy(BaseFreeEnergy):
    def __init__(self, top: Union[topology.SingleTopology, topology.DualTopology], label=None, complex_path=None):
        self.top = top
        self.label = label  # TODO: Do we need these?
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
        Prepares the host-guest system

        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params
        host_system: Optional[openmm.System]
            openmm System object to be deserialized.

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses_a = get_mol_masses(self.mol_a)
        ligand_masses_b = get_mol_masses(self.mol_b)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)

        combined_masses = self._combine(ligand_masses_a, ligand_masses_b, host_masses)
        return final_potentials, final_params, combined_masses

    def prepare_vacuum_edge(self, ff_params):
        """
        Prepares the vacuum system

        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses_a = get_mol_masses(self.mol_a)
        ligand_masses_b = get_mol_masses(self.mol_b)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, self.top)
        combined_masses = self._combine(ligand_masses_a, ligand_masses_b)
        return final_potentials, final_params, combined_masses

    def prepare_combined_coords(self, host_coords=None):
        """
        Returns the combined coordinates.

        Parameters
        ----------
        host_coords: Optional[np.array]
            Nx3 array of atomic coordinates.
            If None, return just the combined ligand coordinates.

        Returns
        -------
            combined_coordinates
        """
        ligand_coords_a = get_romol_conf(self.mol_a)
        ligand_coords_b = get_romol_conf(self.mol_b)

        return self._combine(ligand_coords_a, ligand_coords_b, host_coords)

    def _combine(self, ligand_values_a, ligand_values_b, host_values=None):
        """
        Combine the values along the 0th axis. The host values will be first, if given.
        The ligand values will be combined in a way that matches the topology.
        For single topology this means interpolating the ligand_values_a and
        ligand_values_b. For dual topology this is just concatenation.

        Returns
        -------
            combined_values
        """
        if isinstance(self.top, topology.SingleTopology):
            ligand_values = [np.mean(self.top.interpolate_params(ligand_values_a, ligand_values_b), axis=0)]
        else:
            ligand_values = [ligand_values_a, ligand_values_b]
        all_values = [host_values] + ligand_values if host_values is not None else ligand_values
        return np.concatenate(all_values)
