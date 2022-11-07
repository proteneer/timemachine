from simtk import unit

BOLTZMANN = 1.380658e-23  # J/kelvin
AVOGADRO = 6.0221367e23  # mol^-1
RGAS = BOLTZMANN * AVOGADRO  # J/mol per kelvin
BOLTZ = RGAS / 1000  # kJ/mol per kelvin
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79  # http://openmopac.net/manual/Hessian_Matrix.html

ENERGY_UNIT = unit.kilojoule_per_mole
DISTANCE_UNIT = unit.nanometer

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
DEFAULT_FF = "smirnoff_1_1_0_ccc.py"
DEFAULT_TEMP = 300  # kelvin
DEFAULT_KT = BOLTZ * DEFAULT_TEMP  # kJ/mol

KCAL_TO_KJ = 4.184  # multiply to convert from kcal/mol to kJ/mol
KCAL_TO_DEFAULT_KT = KCAL_TO_KJ / DEFAULT_KT

DEFAULT_PROTEIN_FF = "amber99sbildn"
DEFAULT_WATER_FF = "tip3p"
PROTEIN_FF_TAG = "ProteinForcefield"
WATER_FF_TAG = "WaterForcefield"


# used to check norms in the gradient computations
MAX_FORCE_NORM = 25000
