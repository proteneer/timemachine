import unittest
import numpy as np
from timemachine.constants import ONE_4PI_EPS0

class ReferenceEwaldEnergy():

    def __init__(self, params, param_idxs, box, exclusions):
        self.params = params
        self.param_idxs = param_idxs # length N
        self.charges = [self.params[p_idx] for p_idx in self.param_idxs]
        self.num_atoms = len(self.charges)
        self.box = box # we probably want this to be variable when we start to work on barostats
        self.exclusions = exclusions
        self.alphaEwald = 1.0

    def energy(self, conf):
        return self.reciprocal_energy(conf)

    def reciprocal_energy(self, conf):
        # Reference implementation taken from ReferenceLJCoulombIxn.cpp in OpenMM
        N = self.num_atoms
        recipBoxSize = np.array([(2*np.pi)/self.box[0], (2*np.pi)/self.box[1], (2*np.pi)/self.box[2]])
        kmax = 10
        eir = np.zeros((kmax, N, 3), dtype=np.complex128)
        tab_xy = np.zeros(N, np.complex128)
        tab_qxyz = np.zeros(N, np.complex128)

        totalRecipEnergy = 0

        for i in range(N):
            coords = conf[i, :]

            # j == 0
            for m in range(3):
                eir[0, i, m] = np.complex(1., 0.)

            # j == 1
            for m in range(3):
                eir[1, i, m] = np.complex(
                    np.cos(coords[m]*recipBoxSize[m]), # real
                    np.sin(coords[m]*recipBoxSize[m])  # imag
                )
            # j == 2
            for j in range(2, kmax):
                for m in range(3):
                    eir[j, i, m] = eir[j-1, i, m] * eir[1, i, m]

        print(eir)

        lowry = 0
        lowrz = 1

        numRx = kmax
        numRy = kmax
        numRz = kmax

        factorEwald = -1 / (4*self.alphaEwald*self.alphaEwald)
        epsilon = 1.0
        recipCoeff = ONE_4PI_EPS0*4*np.pi/(self.box[0]*self.box[1]*self.box[2])/epsilon;

        for rx in range(numRx):

            kx = rx * recipBoxSize[0]

            for ry in range(lowry, numRy):

                ky = ry * recipBoxSize[1]

                if ry >= 0:
                    for n in range(N):
                        tab_xy[n] = eir[rx, n, 0] * eir[ry, n, 1]
                else:
                    for n in range(N):
                        tab_xy[n] = eir[rx, n, 0] * np.conj(eir[-ry, n, 1])

                for rz in range(lowrz, numRz):
                    # print(rx, ry, rz)
                    if rz >= 0:
                        for n in range(N):
                            tab_qxyz[n] = self.charges[i] * tab_xy[n] * eir[rz, n, 2]
                    else:
                        for n in range(N):
                            tab_qxyz[n] = self.charges[i] * tab_xy[n] * np.conj(eir[-rz, n, 2])

                    cs = 0
                    ss = 0

                    for n in range(N):
                        cs += tab_qxyz[n].real
                        ss += tab_qxyz[n].imag

                    kz = rz * recipBoxSize[2]
                    k2 = kx * kx + ky*ky + kz*kz
                    ak = np.exp(k2*factorEwald) / k2

                    recipEnergy = recipCoeff * ak * (cs * cs + ss * ss)

                    totalRecipEnergy += recipEnergy

                    lowrz = 1 - numRz

                lowry = 1 - numRy

        return totalRecipEnergy


class TestPeriodicForce(unittest.TestCase):

    def test_reference_ewald(self):

        # do the hard non-convergent loop

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        exclusions = np.array([
            [0,0,1,0,0],
            [0,0,0,1,1],
            [1,0,0,0,0],
            [0,1,0,0,1],
            [0,1,0,1,0],
            ], dtype=np.bool)

        box = [10.0, 10.0, 10.0]

        params = np.array([1.3, 0.3], dtype=np.float64)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)

        ref = ReferenceEwaldEnergy(params, param_idxs, box, exclusions)
        print(ref.energy(x0))


if __name__ == "__main__":
    unittest.main()


