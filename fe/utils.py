import numpy as np
import simtk.unit

def set_velocities_to_temperature(n_atoms, temperature, masses):
    assert 0 # don't call this yet
    v_t = np.random.normal(size=(n_atoms, 3))
    velocity_scale = np.sqrt(constants.BOLTZ*temperature/np.expand_dims(masses, -1))
    return v_t*velocity_scale

def compute_d2e_dxdp(energies, params, xs, dp_idxs):
    mixed_partials = []
    for p in energies:
        _, _, ph, _, pmp  = p.derivatives(np.expand_dims(xs, axis=0), params, dp_idxs)
        mixed_partials.append(pmp)

    mixed_part = np.sum(mixed_partials, axis=0)[0]
    return mixed_part

def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)


def write(xyz, masses):
    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + '\n'
    buf += 'timemachine\n'
    for m, (x,y,z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = 'C'
        elif int(round(m)) == 14:
            symbol = 'N'
        elif int(round(m)) == 16:
            symbol = 'O'
        elif int(round(m)) == 32:
            symbol = 'S'
        elif int(round(m)) == 35:
            symbol = 'Cl'
        elif int(round(m)) == 1:
            symbol = 'H'
        elif int(round(m)) == 31:
            symbol = 'P'
        elif int(round(m)) == 19:
            symbol = 'F'
        elif int(round(m)) == 80:
            symbol = 'Br'
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + ' ' + str(round(x,5)) + ' ' + str(round(y,5)) + ' ' +str(round(z,5)) + '\n'
    return buf