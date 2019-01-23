import numpy as np
import os
import unittest
import tensorflow as tf

from timemachine.functionals import bonded
from timemachine.functionals import gbsa
from timemachine.derivatives import densify
# OpenMM nonbonded terms
# NoCutoff = 0,
# CutoffNonPeriodic = 1,
# CutoffPeriodic = 2,
# Ewald = 3,
# PME = 4,
# LJPME = 5

import xml.etree.ElementTree as ET


def deserialize_system(xml_file):
    """
    Deserialize an openmm XML file into a set of functional forms
    supported by the time machine.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    masses = []

    all_nrgs = []

    for child in root:
        if child.tag == 'Particles':
            for subchild in child:
                masses.append(np.float64(subchild.attrib['mass']))
        if child.tag == 'Forces':
            for subchild in child:
                tags = subchild.attrib
                force_group = np.int32(tags['forceGroup'])
                force_version = np.int32(tags['version'])
                force_type = tags['type']
                if force_type == 'HarmonicBondForce':
                    force_periodic = tags['usesPeriodic']

                    params = []
                    param_idxs = []
                    bond_idxs = []

                    for bonds in subchild:
                        for bond in bonds:
                            d = np.float64(bond.attrib['d'])
                            k = np.float64(bond.attrib['k'])
                            src = np.int32(bond.attrib['p1'])
                            dst = np.int32(bond.attrib['p2'])

                            p_idx_d = len(params)
                            params.append(d)
                            p_idx_k = len(params)
                            params.append(k)

                            param_idxs.append((p_idx_k, p_idx_d))
                            bond_idxs.append((src, dst))

                    params = np.array(params)
                    param_idxs = np.array(param_idxs)
                    bond_idxs = np.array(bond_idxs)

                    all_nrgs.append(bonded.HarmonicBond(params, bond_idxs, param_idxs))
                elif force_type == 'HarmonicAngleForce':
                    print('HarmonicAngleForce not fully implemented yet')

                elif force_type == 'PeriodicTorsionForce':

                    params = []
                    param_idxs = []
                    torsion_idxs = []

                    for tors in subchild:
                        for tor in tors:
                            k = np.float64(tor.attrib['k'])
                            phase = np.float64(tor.attrib['phase'])
                            periodicity = np.float64(tor.attrib['periodicity'])
                            p1 = np.int32(tor.attrib['p1'])
                            p2 = np.int32(tor.attrib['p2'])
                            p3 = np.int32(tor.attrib['p3'])
                            p4 = np.int32(tor.attrib['p4'])

                            p_idx_k = len(params)
                            params.append(k)
                            p_idx_phase = len(params)
                            params.append(phase)
                            p_idx_period = len(params)
                            params.append(periodicity)

                            param_idxs.append((p_idx_k, p_idx_phase, p_idx_period))
                            torsion_idxs.append((p1, p2, p3, p4))

                    params = np.array(params)
                    param_idxs = np.array(param_idxs)
                    torsion_idxs = np.array(torsion_idxs)
                    all_nrgs.append(bonded.PeriodicTorsion(params, torsion_idxs, param_idxs))


                # elif force_type == 'HarmonicAngleForce':
                #     force_periodic = tags['usesPeriodic']


                # elif force_type == 'NonbondedForce':
                #     method = np.int32(tags['method'])

                #     if method != 0 and method != 1:
                #         raise TypeError('Only nonperiodic systems are supported for now.')

                #     alpha = np.float64(tags['alpha'])
                #     cutoff = np.float64(tags['cutoff'])
                #     dispersionCorrection = np.bool(tags['dispersionCorrection'])
                #     ewaldTolerance = np.float64(tags['ewaldTolerance'])
                #     ljAlpha = np.float64(tags['ljAlpha'])



                #     : '0', 'ljnx': '0', 'ljny': '0', 'ljnz': '0', 'method': '1', 'nx': '0', 'ny': '0', 'nz': '0', 'recipForceGroup': '-1', 'rfDielectric': '1', 'switchingDistance': '-1', 'type': 'NonbondedForce', 'useSwitchingFunction': '0', 'version': '3'}
                elif force_type == 'GBSAOBCForce':

                    soluteDielectric = np.float64(tags['soluteDielectric'])
                    solventDielectric = np.float64(tags['solventDielectric'])
                    surfaceAreaEnergy = np.float64(tags['surfaceAreaEnergy'])
                    cutoff = np.float64(tags['cutoff'])
                    # ignore surface area term in non-polar approximation

                    params = []
                    param_idxs = []

                    for particles in subchild:
                        for particle in particles:
                            q_idx = len(params)
                            params.append(np.float64(particle.attrib['q']))
                            r_idx = len(params)
                            params.append(np.float64(particle.attrib['r']))
                            s_idx = len(params)
                            params.append(np.float64(particle.attrib['scale']))

                            param_idxs.append((q_idx, r_idx, s_idx))

                    params = np.array(params)
                    param_idxs = np.array(param_idxs)

                    all_nrgs.append(gbsa.GBSAOBC(
                        params,
                        param_idxs,
                        cutoffDistance=cutoff,
                        soluteDielectric=soluteDielectric,
                        solventDielectric=solventDielectric,
                        surfaceAreaEnergy=surfaceAreaEnergy))

                # else:
                #     raise TypeError("Unsupported force type:", force_type)

                # print(tags)
        # print(child.tag, child.attrib)

    return masses, all_nrgs

def deserialize_state(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    pot_nrg = None
    coords = []
    velocities = []
    forces = []

    for child in root:
        if child.tag == 'Energies':
            pot_nrg = np.float64(child.attrib['PotentialEnergy'])
            # for subchild in child:
                # masses.append(np.float64(subchild.attrib['mass']))
        elif child.tag == 'Positions':
            for subchild in child:
                x, y, z = np.float64(subchild.attrib['x']), np.float64(subchild.attrib['y']), np.float64(subchild.attrib['z'])
                coords.append((x,y,z))
        elif child.tag == 'Velocities':
            for subchild in child:
                x, y, z = np.float64(subchild.attrib['x']), np.float64(subchild.attrib['y']), np.float64(subchild.attrib['z'])
                velocities.append((x,y,z))
        elif child.tag == 'Forces':
            for subchild in child:
                x, y, z = np.float64(subchild.attrib['x']), np.float64(subchild.attrib['y']), np.float64(subchild.attrib['z'])
                forces.append((x,y,z))

    return pot_nrg, np.array(coords), np.array(velocities), np.array(forces)


def get_data(fname):
    return os.path.join(os.path.dirname(__file__), 'data', fname)


class TestAlaAlaAla(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_ala(self):

        masses, nrgs = deserialize_system(get_data('system.xml'))
        ref_nrg, x0, velocities, ref_forces = deserialize_state(get_data('state0.xml'))

        num_atoms = x0.shape[0]

        x_ph = tf.placeholder(shape=(num_atoms, 3), dtype=tf.float64)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        # bonded
        nrg_op = nrgs[0].energy(x_ph)
        grad_op = densify(tf.gradients(nrg_op, x_ph)[0])
        nrg_val, grad_val = sess.run([nrg_op, grad_op], feed_dict={x_ph: x0})
        np.testing.assert_almost_equal(ref_nrg, nrg_val)
        np.testing.assert_almost_equal(ref_forces, grad_val*-1)

        # torsion
        ref_nrg, x0, velocities, ref_forces = deserialize_state(get_data('state2.xml'))
        nrg_op = nrgs[1].energy(x_ph)
        grad_op = densify(tf.gradients(nrg_op, x_ph)[0])
        nrg_val, grad_val = sess.run([nrg_op, grad_op], feed_dict={x_ph: x0})
        np.testing.assert_almost_equal(ref_nrg, nrg_val)
        np.testing.assert_almost_equal(ref_forces, grad_val*-1)

        # GBSA
        ref_nrg, x0, velocities, ref_forces = deserialize_state(get_data('state4.xml'))
        nrg_op = nrgs[2].energy(x_ph)
        grad_op = densify(tf.gradients(nrg_op, x_ph)[0])
        nrg_val, grad_val = sess.run([nrg_op, grad_op], feed_dict={x_ph: x0})
        np.testing.assert_almost_equal(ref_nrg, nrg_val)
        np.testing.assert_almost_equal(ref_forces, grad_val*-1)



if __name__ == "__main__":
    unittest.main()
        