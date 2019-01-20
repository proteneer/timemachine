import os
import numpy as np
import tensorflow as tf
import pint

ureg = pint.UnitRegistry()

tf.reset_default_graph()

from timemachine.functionals import bonded, nonbonded
from timemachine import integrator
import xmltodict
import time

file_dir = os.path.dirname(__file__)

def get_box_and_conf():
    with open(os.path.join(file_dir, 'water/state.xml')) as fd:
        doc = xmltodict.parse(fd.read())
        box = doc['State']['PeriodicBoxVectors']
        x = np.float64(box['A']['@x'])
        y = np.float64(box['B']['@y'])
        z = np.float64(box['C']['@z'])
        coords = doc['State']['Positions']
        geom = []
        for elem in coords['Position']:
            geom.append((
                np.float64(elem['@x']),
                np.float64(elem['@y']),
                np.float64(elem['@z']))
            )
        return np.array([x,y,z]), np.array(geom)

# epis ii, epis jj: eps: 2.54387

def get_system():
    with open(os.path.join(file_dir, 'water/system.xml')) as fd:
        doc = xmltodict.parse(fd.read())
        sys = doc['System']
        masses = []



        charge_params = tf.convert_to_tensor(np.array([.417, -.834], dtype=np.float64))
        charge_idxs = []

        # oxygen, H

        # lj_params = tf.convert_to_tensor(np.array([.3150752406575124, .635968, 1, 0], dtype=np.float64))
        # 1.59495
        lj_params = tf.convert_to_tensor(np.array([.3150752406575124/2, 1.59495, 1.0/2, 0.0], dtype=np.float64))
        lj_idxs = []
        for p in sys['Particles']['Particle']:
            mass = np.float64(p['@mass'])
            masses.append(mass)
            if mass > 2:
                charge_idxs.append(1)
                lj_idxs.append((0, 1))
            else:
                charge_idxs.append(0)
                lj_idxs.append((2, 3))

        lj_idxs = np.array(lj_idxs)
        print(lj_idxs)
            
        masses = np.array(masses)
        num_atoms = masses.shape[0]
    
        bond_params = [
            tf.get_variable(name='bond_k', dtype=tf.float64, shape=tuple(), initializer=tf.constant_initializer(462750.4)),
#             tf.get_variable(name='bond_k', dtype=tf.float64, shape=tuple(), initializer=tf.constant_initializer(6000.4)),
            tf.get_variable(name='bond_d', dtype=tf.float64, shape=tuple(), initializer=tf.constant_initializer(0.09572)),
        ]
        bond_idxs = []
        bond_param_idxs = []

        angle_params = [
            tf.get_variable(name='angle_k', dtype=tf.float64, shape=tuple(), initializer=tf.constant_initializer(836.8)),
#             tf.get_variable(name='angle_k', dtype=tf.float64, shape=tuple(), initializer=tf.constant_initializer(600.8)),
            tf.get_variable(name='angle_theta', dtype=tf.float64, shape=tuple(), initializer=tf.constant_initializer(1.82421813418)),
        ]
        angle_idxs = []
        angle_param_idxs = []

        exclusions = np.zeros(shape=(num_atoms, num_atoms), dtype=np.bool)
        
        for f in sys['Forces']['Force']:
            if f['@type'] == 'HarmonicBondForce':
                for b in f['Bonds']['Bond']:
                    src, dst = np.int64(b['@p1']), np.int64(b['@p2'])
                    bond_idxs.append((src, dst))
                    exclusions[src][dst] = 1
                    exclusions[dst][src] = 1
                    bond_param_idxs.append((0, 1))
            if f['@type'] == 'HarmonicAngleForce':
                for a in f['Angles']['Angle']:
                    src, mid, dst = np.int64(a['@p1']), np.int64(a['@p2']), np.int64(a['@p3'])
                    angle_idxs.append((src, mid, dst))

                    # commented out because they're implied
                    assert exclusions[src][mid] == 1
                    assert exclusions[mid][src] == 1
                    assert exclusions[mid][dst] == 1
                    assert exclusions[dst][mid] == 1
                    exclusions[src][dst] = 1
                    exclusions[dst][src] = 1
                    
                    angle_param_idxs.append((0, 1))

        bond_idxs = np.array(bond_idxs)
        bond_param_idxs = np.array(bond_param_idxs)

        angle_idxs = np.array(angle_idxs)
        angle_param_idxs = np.array(angle_param_idxs)

        hb = bonded.HarmonicBond(bond_params, bond_idxs, bond_param_idxs)
        ha = bonded.HarmonicAngle(angle_params, angle_idxs, angle_param_idxs)
        es = nonbonded.Electrostatic(charge_params, charge_idxs, exclusions)
        lj = nonbonded.LeonnardJones(lj_params, lj_idxs, exclusions)
        return masses, [hb, ha, es, lj]

def make_xyz(masses, coords):
    num_atoms = coords.shape[0]
    res = str(num_atoms) + "\n"
    res += "\n"

    for idx in range(num_atoms):
        if masses[idx] > 2:
            element = "O"
        else:
            element = "H"
        c = coords[idx]
        res += element + " " + str(c[0]*10) + " " + str(c[1]*10) + " " + str(c[2]*10) + "\n"

    return res
    

box, x0 = get_box_and_conf()
masses, energies = get_system()

total_mass = 0
for m in masses:
    total_mass += m * ureg.amu

def density(box):
    raw = total_mass.to('kg')/((box[0]*ureg.nm) * (box[1]*ureg.nm) * (box[2]*ureg.nm))
    return raw.to(ureg.kg/(ureg.meter*ureg.meter*ureg.meter))


num_atoms = x0.shape[0]

x_ph = tf.placeholder(name="x", shape=(num_atoms, 3), dtype=tf.float64)

friction = 1.0
dt = 0.002
temp = 300

box_ph = tf.placeholder(shape=(3,), dtype=np.float64)

intg = integrator.LangevinIntegrator(masses, x_ph, box_ph, energies, dt, friction, temp)
dx_op, db_op = intg.step_op(inference=True)

num_steps = 500000

sess = tf.Session()
sess.run(tf.initializers.global_variables())

x = x0.copy()
b = box.copy()
all_xyz = ""
s_time = time.time()
for step in range(1000000):
    dx_val, db_val, db_base, tot_E, e_pv, e_NRT = sess.run([dx_op, db_op, intg.dE_db, intg.all_Es, intg.pv, intg.NRT], feed_dict={x_ph: x, box_ph: b})
    if step % 100 == 0 or step < 100:
        print("step", step, "box", b, "volume", np.prod(b), "density", density(b), ", ns/day", (step * dt * 86400) / ((time.time() - s_time) * 1000), "E:", tot_E, 'E_pv', e_pv, 'E_NRT', e_NRT, 'E_free', tot_E + e_pv - e_NRT)
        all_xyz += make_xyz(masses, x)

        # DEBUG
        # with open("frames.xyz", "w") as fd:
            # fd.write(all_xyz)
        if step % 10000 == 0:
            print("checkpointing..")
            with open("frames.xyz", "w") as fd:
                fd.write(all_xyz)      


    x += dx_val
    # do we need to scale the coordinates?
    b += db_val/(num_atoms*100)

with open("frames.xyz", "w") as fd:
    fd.write(all_xyz)

lastModel = None

