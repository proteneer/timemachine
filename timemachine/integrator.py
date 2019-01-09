import numpy as np
import tensorflow as tf

from timemachine.constants import BOLTZ

class LangevinIntegrator():

    def __init__(self,
        masses,
        x_t,
        energies, # ytz TODO: rename to forces
        dt=0.0025,
        friction=1.0,
        temp=300.0,
        precision=tf.float64,
        buffer_size=None):
        """
        Langevin Integrator is an implementation of stochastic dynamics that samples
        from a gaussian distribution to simulate the effects of a thermostat.

        Parameters
        ----------
        masses: (N,) list of float-like objects
            Masses of each atom in the system

        x_t: (N, 3) tf.placeholder
            Used to feed in the geometry of the system

        energies: list of timemachine.ConservativeForces
            Various energies of the system

        dt: float
            Time step in nanoseconds

        friction: float
            Dissipation or memory-losing ability of the heat bath

        temp: float
            Temperature used for drawing random velocities. Is this is zero then no
            random noise will be added.

        This is inspired by ReferenceStochasticDynamics.cpp from OpenMM.

        """
        self.x_t = x_t
        self.dt = dt
        self.friction = friction # dissipation (how fast we forget)
        self.temperature = temp  # temperature
        self.vscale = np.exp(-self.dt*self.friction)

        if self.friction == 0:
            self.fscale = self.dt
        else:
            self.fscale = (1-self.vscale)/self.friction

        kT = BOLTZ * self.temperature
        self.nscale = np.sqrt(kT*(1-self.vscale*self.vscale)) # noise scale
        self.normal = tf.distributions.Normal(loc=np.float64(0.0), scale=np.float64(1.0))
        # print(self.normal.dtype)
        # assert 0
        self.invMasses = (1.0/masses).reshape((-1, 1))
        self.sqrtInvMasses = np.sqrt(self.invMasses)

        self.ca = tf.cast(self.vscale, dtype=precision)
        self.cbs = []
        for m in masses:
            self.cbs.append((self.fscale*self.dt)/m)

        self.cbs = tf.convert_to_tensor(self.cbs, dtype=precision)
        self.cbs = tf.reshape(self.cbs, shape=(1, -1, 1))

        if buffer_size is None:
            buffer_size = 4000 # guestimate later

        # pre-compute scaled prefactors once at the beginning
        self.scale = (1-tf.pow(self.ca, tf.range(buffer_size, dtype=precision)+1))/(1-self.ca)
        self.scale = tf.reverse(self.scale, [0])
        self.scale = tf.reshape(self.scale, (-1, 1, 1, 1))

        self.num_atoms = len(masses)
        self.energies = energies

        self.num_params = sum([e.total_params() for e in energies])
        
        self.v_t = tf.get_variable(
            "buffer_velocity",
            shape=(self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # buffer for current step's dxdp
        self.dxdp_t = tf.get_variable(
            "buffer_dxdp",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # buffer for unconverged zetas
        self.buffer_zetas = tf.get_variable(
            "buffer_zetas",
            shape=(buffer_size, self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # buffer for converged zetas
        self.converged_zetas = tf.get_variable(
            "converged_zetas",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # compute necessary derivatives:
        gs = []
        hs = []
        es = []
        for e in self.energies:
            es.append(e.energy(x_t))
            gs.append(e.gradients(x_t))
            hs.append(e.hessians(x_t))

        self.tot_e = tf.reduce_sum(tf.stack(es))
        self.grads = tf.reduce_sum(tf.stack(gs), axis=0)
        self.hessians = tf.reduce_sum(tf.stack(hs), axis=0)

        mixed_partials = []

        self.mixed_partial_shapes = []
        for e_idx, e in enumerate(self.energies):
            parameter_shapes = []
            for mp_idx, mp in enumerate(e.mixed_partials(x_t)):
                # store the shapes
                p_shapes = mp.get_shape().as_list()
                parameter_shapes.append(p_shapes)

                # compute the mixed partial derivatives
                mp_N, mp_D = p_shapes[-2], p_shapes[-1]
                flattened = tf.reshape(mp, (-1, mp_N, mp_D))
                mixed_partials.append(flattened)

            self.mixed_partial_shapes.append(parameter_shapes)

        self.mixed_partials = tf.concat(mixed_partials, axis=0) # [num_params, num_atoms, 3]

    def reset(self, session):
        """
        Reset the state of the integrator using a session.
        """
        session.run([
            self.v_t.initializer,
            self.dxdp_t.initializer,
            self.buffer_zetas.initializer,
            self.converged_zetas.initializer
        ])

    def jacs_and_vars(self):
        """
        Returns a list of (jacobians, variable) at some geometry x

        Params
        ------
        dLdx: (N,3) tf.Tensor
            Derivative of the observable with respect to the geometry.

        Returns
        -------
        tuples of jacobians (p0, p1, ..., N, 3) and vars (p0,p1,...)
            The grads correspond to the gradient of some loss function with respect
            to the parameters.

        """
        jvs = []

        # indices are used to keep track of start and end indices
        start_idx = 0
        end_idx = 0
        for e_idx, nrg in enumerate(self.energies):
            for p_idx, p in enumerate(nrg.get_params()):
                cur_shape = self.mixed_partial_shapes[e_idx][p_idx]
                tot_size = np.prod(cur_shape[:-2], dtype=np.int32) # last two dims are Nx3
                end_idx += tot_size
                jacs = self.dxdp_t[start_idx:end_idx]

                jvs.append((jacs, p))
                start_idx += tot_size

        return jvs

    def grads_and_vars(self, dLdx):
        """
        Params
        ------
        dLdx: (N, 3) tf.Tensor
            reduced derivatives of the loss with respect to geometry

        """
        # reduce the jacobian by broadcasting observable loss

        gvs = []
        for jac, param in self.jacs_and_vars():
            # (ytz): We can probably optimize away two of these reshapes, but
            # they should be relatively cheap so whatever.
            dLdx_dxdp = tf.multiply(tf.reshape(dLdx, jac.get_shape()), jac)
            dxdp = tf.reduce_sum(dLdx_dxdp, axis=[-1, -2])
            gv = (tf.reshape(dxdp, param.get_shape()), param)
            gvs.append(gv)

        return gvs

    def step_op(self, inference=False):
        """
        Generate ops that propagate the time by one step.

        Parameters
        ----------
        inference: bool
            If inference is True then we disable generation of extra derivatives needed for training.

        Returns
        -------
        tuple: (tf.Op, tf.Op)
            Returns two operations required to advance the time step. Both must be run in a tf.Session.
            If inference is True then the second element is None.

        This should be run exactly once. Remember to call reset() afterwards.

        """
        x_t = self.x_t
        tot_e = self.tot_e
        grads = self.grads
        hessians = self.hessians

        num_dims = 3
        num_atoms = self.num_atoms

        noise = self.normal.sample((num_atoms, num_dims))

        new_v_t = self.vscale*self.v_t - self.fscale*self.invMasses*grads + self.nscale*self.sqrtInvMasses*noise
        v_t_assign = tf.assign(self.v_t, new_v_t)

        # compute dxs
        dx = v_t_assign * self.dt

        if inference:
            return dx, None

        # The Algorithm:

        # 1. Given starting geometry and velocity x_t, v_t, dxdp_t, zetas_[0, t)
        # 2. Compute grads, hessians, and mixed partials: g_t, h_t, mp_t
        # 3. Using previous zetas_[0, t), first compute zeta_t for time t using h_t, mp_t, dxdp_t to compute zeta_t, appending
        #    zeta_t to list of zetas
        # 4. Return new geometry and dxdp

        # compute dxdp, derivative of the geometry with respect to parameters
        # This uses only [0, t) and doesn't require zeta of the current step

        mixed_partials = self.mixed_partials
        # Rolling Algorithm, insertion is at the front, highest exponent is at the front:
        # buf size: 5         buf   start_idx
        
        # step 0: [-.-,-,-,0] -     0
        # step 1: [-,-,-,0,1] -     0
        # step 2: [-,-,0,1,2] -     0
        # step 3: [-,0,1,2,3] -     0
        # step 4: [0,1,2,3,4] -     0
        # step 5: [1,2,3,4,5] 0     1
        # step 6: [2,3,4,5,6] 0+1   2
        # step 7: [3,4,5,6,7] 0+1+2 3

        # 1. Accumulate first element in buffer into converged zetas
        # 2. Insert to the front
        # 3. Cyclic-roll buffer left

        # compute zeta_t for time t
        # a:  0 1 2 3            0 1 2
        # h: [N,3,N,3], dxdp_t: [p,N,3], contraction: [p,N,3], mp: [p,N,3]
        contraction = tf.einsum('ijkl,mkl->mij', hessians, self.dxdp_t)
        zeta_t = contraction + mixed_partials

        converged_zetas_assign = tf.assign_add(self.converged_zetas, self.buffer_zetas[0])
        with tf.control_dependencies([self.converged_zetas, self.dxdp_t]):
            override = self.buffer_zetas[0].assign(zeta_t)
            buffer_zetas_assign = tf.assign(
                self.buffer_zetas,
                tf.roll(override, shift=-1, axis=0)
            )

            # compute dxdp_{t+1} using unconverged zeta and converged zetas
            new_dxdp_t = buffer_zetas_assign * self.scale
            new_dxdp_t = tf.reduce_sum(new_dxdp_t, axis=0)
            new_dxdp_t += converged_zetas_assign * self.scale[0][0][0][0]
            new_dxdp_t *= -self.cbs

            dxdp_t_assign = tf.assign(self.dxdp_t, new_dxdp_t)

            # (ytz): note we *MUST* return self.dxdp_t for this not to have a dependency hell
            return dx, dxdp_t_assign
