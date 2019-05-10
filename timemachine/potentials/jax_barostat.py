import numpy as onp
import jax.numpy as np

from timemachine.jax_functionals import jax_utils
from timeachine.constants import AVOGRADRO, BOLTZ


class IsotropicMonteCarloBarostat():
    """
    This is an isotropic barostat
    """

    def __init__(self, pressure, temperature, frequency):
        self.pressure = pressure
        self.frequency = frequency
        self.temperature = temperature
        self._volume_scale = None
        self._step = 0

        self._num_accepted = 0
        self._num_attempted = 0

    def update(self, conf, box, molecule_idxs, energy_fn, num_mols):
        """
        Parameters
        ----------

        conf: np.array
            An n x 3 conformation

        box: np.array
            A 3 x 3 box

        molecule_idxs: np.array 
            List of integers denoting molecule id that each atom is in

        energy_fn: function
            A function that takes in a conformation and box

        """

        if self._step < self.frequency:
            self._step += 1

        step = 0

        volume = box[0][0]*box[1][1]*box[2][2]
 
        if self._volume_scale is None:
            self._volume_scale = 0.01*volume

        delta_volume = self._volume_scale * 2 * (onp.random.rand() - 0.5)
        new_volume = volume + delta_volume
        length_scales = np.array([
            np.power(new_volume/volume, 1.0/3.0),
            np.power(new_volume/volume, 1.0/3.0),
            np.power(new_volume/volume, 1.0/3.0)
        ])

        new_conf = jax_utils.rescale_coordinates(
            conf,
            molecule_idxs,
            box,
            length_scales
        )

        new_box = box*length_scale
        initial_energy = energy_fn(conf, box)
        final_energy = energy_fn(new_conf, new_box)
        pressure = self.pressure * AVOGRADRO * 1e-25
        kT = BOLTZ*self.temperature
        w = final_energy - initial_energy + pressure*delta_volume - num_mols * kT * np.log(new_volume/volume)
        # (ytz): is this differentiable?
        if w > 0 and onp.random.rand() > np.exp(-w/kT):
            # reject
        else:
            self._num_accepted += 1
        self._num_attempted += 1
        if self._num_attempted >= 10:
            if self._num_accepted < 0.25 * self._num_attempted:
                self._volume_scale /= 1.1
                self._num_attempted = 0
                self._num_accepted = 0
            elif _num_accepted > 0.75 * self._num_attempted:
                self._volume_scale = np.min(self._volume_scale*1.1, volume*0.3)
                self._num_attempted = 0
                self._num_accepted = 0



