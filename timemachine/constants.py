from typing import Any, Dict

# physical constants
BOLTZMANN = 1.380658e-23  # J/kelvin
AVOGADRO = 6.0221367e23  # mol^-1
RGAS = BOLTZMANN * AVOGADRO  # J/mol per kelvin
BOLTZ = RGAS / 1000  # kJ/mol per kelvin
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79  # http://openmopac.net/manual/Hessian_Matrix.html

# default thermodynamic ensemble
DEFAULT_TEMP = 300.0  # kelvin
DEFAULT_PRESSURE = 1.013  # bar
DEFAULT_KT = BOLTZ * DEFAULT_TEMP  # kJ/mol

# common unit conversions
BAR_TO_KJ_PER_NM3 = 1e-25  # kJ/nm^3
KCAL_TO_KJ = 4.184  # multiply to convert from kcal/mol to kJ/mol
KCAL_TO_DEFAULT_KT = KCAL_TO_KJ / DEFAULT_KT

# default force fields
DEFAULT_FF = "smirnoff_2_0_0_ccc.py"
DEFAULT_PROTEIN_FF = "amber99sbildn"
DEFAULT_WATER_FF = "amber14/tip3p"

DEFAULT_CHIRAL_ATOM_RESTRAINT_K = 1000.0
DEFAULT_CHIRAL_BOND_RESTRAINT_K = 999.9

# thresholds
MAX_FORCE_NORM = 50000.0  # used to check norms in the gradient computations

# atom mapping parameters
DEFAULT_ATOM_MAPPING_KWARGS: Dict[str, Any] = {
    "ring_cutoff": 0.12,
    "chain_cutoff": 0.2,
    "max_visits": 1e7,
    "max_connected_components": 1,
    "min_connected_component_size": 1,
    "max_cores": 1e5,
    "enforce_core_core": True,
    "ring_matches_ring_only": True,
    "enforce_chiral": True,
    "disallow_planar_torsion_flips": True,
    "min_threshold": 0,
    "initial_mapping": None,
}
