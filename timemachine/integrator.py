import numpy as np
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
import inspect


from timemachine.derivatives import list_jacobian
from timemachine.constants import BOLTZ

class LangevinIntegrator():

    def __init__(self,
        masses,
        x_t,
        b_t,
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

        b_t: (3,) or None
            If not None then we expect this to be a rectangular-size lattice.

        energies: list of timemachine.ConservativeForces
            Various energies of the system

        dt: float
            Time step in nanoseconds

        friction: float
            Dissipation or memory-losing ability of the heat bath

        temp: float
            Temperature used for drawing random velocities. Is this is zero then no
            random noise will be added.

        precision: precision
            Either tf.float32 or tf.float64

        buffer_size: int
            If None then we estimate the buffer size required for convergence automatically.
            Otherwise we try and compute the buffer analytically.


        This is inspired by ReferenceStochasticDynamics.cpp from OpenMM.

        """
        self.x_t = x_t
        self.b_t = b_t
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
        self.invMasses = (1.0/masses).reshape((-1, 1))
        self.sqrtInvMasses = np.sqrt(self.invMasses)

        self.ca = tf.cast(self.vscale, dtype=precision)
        self.cbs = []
        for m in masses:
            self.cbs.append((self.fscale*self.dt)/m)

        self.cbs = tf.convert_to_tensor(self.cbs, dtype=precision)
        self.cbs = tf.reshape(self.cbs, shape=(1, -1, 1))

        if buffer_size is None:
            epsilon = 1e-14
            buffer_size = np.int64(np.log(epsilon)/np.log(self.vscale)+1)
            print("Setting buffer_size to:", buffer_size)

        # pre-compute scaled prefactors once at the beginning
        self.scale = (1-tf.pow(self.ca, tf.range(buffer_size, dtype=precision)+1))/(1-self.ca)
        self.scale = tf.reverse(self.scale, [0])
        self.scale = tf.reshape(self.scale, (-1, 1, 1, 1))
        self.num_atoms = len(masses)
        self.energies = energies

        self.num_params = sum([e.total_params() for e in energies])
        
        self.initializers = []

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
        self.initializers.append(self.dxdp_t.initializer)

        # buffer for unconverged Dps
        self.buffer_Dx = tf.get_variable(
            "buffer_Dx",
            shape=(buffer_size, self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)
        self.initializers.append(self.buffer_Dx.initializer)

        # buffer for converged Dps
        self.converged_Dps = tf.get_variable(
            "converged_Dps",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)
        self.initializers.append(self.converged_Dps.initializer)

        all_Es = []
        all_dE_dx = []
        all_d2E_dx2 = []
        all_d2E_dxdp = []

        x_t_shape = x_t.get_shape().as_list()
        num_atoms = x_t_shape[0]

        if b_t is not None:
            # buffer for current step's dxdp
            b_t_shape = b_t.get_shape().as_list()
            self.dbdp_t = tf.get_variable(
                "buffer_dbdp",
                shape=[self.num_params] + b_t_shape,
                dtype=precision,
                initializer=tf.initializers.zeros)
            self.initializers.append(self.dbdp_t.initializer)

            # buffer for converged Dps
            self.converged_Dps_box = tf.get_variable(
                "converged_Dps_box",
                shape=[self.num_params] + b_t_shape,
                dtype=precision,
                initializer=tf.initializers.zeros)
            self.initializers.append(self.converged_Dps_box.initializer)

            all_dE_db = []
            all_d2E_db2 = []
            all_d2E_dbdp = []

            all_d2E_dxdb = []
            all_d2E_dbdx = []

        # (ytz): I don't like this one bit but it works for now.
        def supports_box(e):
            return 'box' in inspect.getargspec(e.energy)[0]

        self.param_shapes = []

        for nrg in self.energies:
            
            if supports_box(nrg) and b_t is not None:
                E = nrg.energy(x_t, b_t)
            else:
                E = nrg.energy(x_t)

            all_Es.append(E)
            dE_dx = tf.gradients(E, x_t)[0]

            offs = []

            if isinstance(nrg.params, tf.Tensor):
                n_params = [nrg.params]
            else:
                n_params = nrg.params

            for r in list_jacobian(dE_dx, n_params):
                r_shape = tf.reshape(r, shape=(-1, num_atoms, 3))
                offs.append(r.get_shape().as_list())
                all_d2E_dxdp.append(r_shape)
            self.param_shapes.append(offs)

            all_dE_dx.append(dE_dx)
            all_d2E_dx2.append(tf.hessians(E, x_t)[0])

            if supports_box(nrg) and b_t is not None:
                dE_db = tf.gradients(E, b_t)[0]
                all_dE_db.append(dE_db)
                all_d2E_db2.append(tf.hessians(E, b_t)[0])
                all_d2E_dxdb.append(jacobian(dE_dx, b_t, use_pfor=False)) # this uses the tf jacobian
                all_d2E_dbdx.append(jacobian(dE_db, x_t, use_pfor=False)) # this uses the tf jacobian

                for r in list_jacobian(dE_db, n_params): # ? ADJUST
                    all_d2E_dbdp.append(tf.reshape(r, shape=(-1, 3))) # (ytz): needs to be changed for 3x3 box sizes!

        self.all_Es = tf.reduce_sum(tf.stack(all_Es), axis=0) # scalar
        self.dE_dx = tf.reduce_sum(tf.stack(all_dE_dx), axis=0) # [N, 3]
        self.d2E_dx2 = tf.reduce_sum(tf.stack(all_d2E_dx2), axis=0) # [N, 3, N, 3]
        self.d2E_dxdp = tf.concat(all_d2E_dxdp, axis=0) # [p, N, 3]

        if b_t is not None:
            self.dE_db = tf.reduce_sum(tf.stack(all_dE_db), axis=0)
            self.d2E_db2 = tf.reduce_sum(tf.stack(all_d2E_db2), axis=0)
            self.d2E_dxdb = tf.reduce_sum(tf.stack(all_d2E_dxdb), axis=0)
            self.d2E_dbdx = tf.reduce_sum(tf.stack(all_d2E_dbdx), axis=0)
            self.d2E_dbdp = tf.concat(all_d2E_dbdp, axis=0)

        # (ytz): Note for implementation purposes, the order of jacobian differentiation
        # actually matters. The jacobian system in tensorflow expects a fixed size
        # tensor for the outputs, while permitting a variable list of tensors for 
        # inputs. This means that we should naturally use the coordinate derivatives
        # as they all have a fixed N x 3 structure, where as the input parameters
        # can take on a variadic list of tensors of varying sizes.
        # d2E_dxdp = []
        # d2E_dbdp = []

        # self.mixed_partial_shapes = []
        # for e_idx, energy in enumerate(self.energies):

        #     # dE_dx
        #     parameter_shapes = []
        #     for mp_idx, mp in enumerate(list_jacobian(all_dE_dx[e_idx], energy.params)):
        #         # store the shapes
        #         p_shapes = mp.get_shape().as_list()
        #         parameter_shapes.append(p_shapes)

        #         # compute the mixed partial derivatives
        #         # mp_N, mp_D = p_shapes[-2], p_shapes[-1]
        #         flattened = tf.reshape(mp, (-1, mp_N, mp_D))
        #         mixed_partials.append(flattened)

        #     if e.supports_box() and b_t is not None:

        #     # self.mixed_partial_shapes.append(parameter_shapes)

        # self.mixed_partials = tf.concat(mixed_partials, axis=0) # [num_params, num_atoms, 3]

    def reset(self, session):
        """
        Reset the state of the integrator using a session.
        """
        session.run(self.initializers)

    def jacs_and_vars(self):
        """
        Computes a list of (jacobians, variable) with respect to geometry. We don't normally
        need dx_dp.

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
                cur_shape = self.param_shapes[e_idx][p_idx]
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
            left = tf.reshape(dLdx, jac.get_shape())
            dLdx_dxdp = tf.multiply(left, jac)
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
        # tot_e = self.tot_e
        # grads = self.grads
        # hessians = self.hessians

        num_dims = 3
        num_atoms = self.num_atoms

        noise = self.normal.sample((num_atoms, num_dims))

        # assert 0
        # print("fscale", self.fscale, "fscale*invMasses", self.fscale*self.invMasses)
        new_v_t = self.vscale*self.v_t - self.fscale*self.invMasses*self.dE_dx + self.nscale*self.sqrtInvMasses*noise
        v_t_assign = tf.assign(self.v_t, new_v_t)

        # compute dxs
        dx = v_t_assign * self.dt
        
        if self.b_t is None:
            dbox = None
        else:
            dbox = -self.dt*self.dE_db

        if inference:
            return dx, dbox

        # The Algorithm:

        # 1. Given starting geometry and velocity x_t, v_t, dxdp_t, Dps_[0, t)
        # 2. Compute grads, hessians, and mixed partials: g_t, h_t, mp_t
        # 3. Using previous Dps_[0, t), first compute Dp_t for time t using h_t, mp_t, dxdp_t to compute Dp_t, appending
        #    Dp_t to list of Dps
        # 4. Return new geometry and dxdp

        # compute dxdp, derivative of the geometry with respect to parameters
        # This uses only [0, t) and doesn't require Dp of the current step

        # mixed_partials = self.mixed_partials
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

        # 1. Accumulate first element in buffer into converged Dps
        # 2. Insert to the front
        # 3. Cyclic-roll buffer left

        # compute Dp_t for time t
        # a:  0 1 2 3            0 1 2
        # h: [N,3,N,3], dxdp_t: [p,N,3], contraction: [p,N,3], mp: [p,N,3]

        Dx_t = tf.einsum('ijkl,mkl->mij', self.d2E_dx2, self.dxdp_t)
        Dx_t += self.d2E_dxdp

        all_ops = []

        control_deps = []

        # If box vectors are not None, then we have periodic boundary conditions that will
        # affect the equations of motion.
        if self.b_t is not None:
            Dx_t += tf.einsum('ijk,mk->mij', self.d2E_dxdb, self.dbdp_t)
            Db_t = tf.einsum('ij,mj->mi', self.d2E_db2, self.dbdp_t)
            Db_t += self.d2E_dbdp
            Db_t += tf.einsum('ijk,mjk->mi', self.d2E_dbdx, self.dxdp_t)

            converged_Db_assign = tf.assign_add(self.converged_Dps_box, Db_t)

            new_dbdp_t = -self.dt*converged_Db_assign

        converged_Dx_assign = tf.assign_add(self.converged_Dps, self.buffer_Dx[0])
        override = self.buffer_Dx[0].assign(Dx_t)
        buffer_Dx_assign = tf.assign(
            self.buffer_Dx,
            tf.roll(override, shift=-1, axis=0)
        )

        # compute dxdp_{t+1} using unconverged Dp and converged Dps
        new_dxdp_t = buffer_Dx_assign * self.scale
        new_dxdp_t = tf.reduce_sum(new_dxdp_t, axis=0)
        new_dxdp_t += converged_Dx_assign * self.scale[0][0][0][0]
        new_dxdp_t *= -self.cbs

        # To avoid a race condition, we want to make sure the code is complete before we return.
        if self.b_t is not None:

            with tf.control_dependencies([new_dxdp_t, new_dbdp_t]):
                dxdp_t_assign = tf.assign(self.dxdp_t, new_dxdp_t)
                dbdp_t_assign = tf.assign(self.dbdp_t, new_dbdp_t)
                return [dx, dxdp_t_assign], [dbox, dbdp_t_assign]

        else:

            # this probably not necessary if there's only one dependency
            with tf.control_dependencies([new_dxdp_t]):
                dxdp_t_assign = tf.assign(self.dxdp_t, new_dxdp_t)
                return [dx, dxdp_t_assign]
