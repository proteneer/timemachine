from simtk import unit

BOLTZMANN = 1.380658e-23
AVOGADRO = 6.0221367e23
RGAS = BOLTZMANN * AVOGADRO
BOLTZ = RGAS / 1000
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79  # http://openmopac.net/manual/Hessian_Matrix.html

ENERGY_UNIT = unit.kilojoule_per_mole
DISTANCE_UNIT = unit.nanometer

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
DEFAULT_FF = "smirnoff_1_1_0_ccc.py"
