#### Charge models
`bootstrap_am1.py` fits a SMIRKS-based model to approximate AM1BCCELF10 charges from `oequacpac`

#### Protocol optimization
`protocol_optimization/optimize_distance_decoupling_1d.py` interpolates between a uniform distribution and `e^{-(LJ(r) + Coulomb(r)}` using a distance-based decoupling path with independently controllable distance offsets for LJ and Coulomb terms `LJ(r + lj_offset(lam, params)) + Coulomb(r + coulomb_offset(lam, params))`. The dependence of these offsets on the scalar control parameter `lam` is optimized to minimize TI variance.

