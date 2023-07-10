#include <cmath>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>

#include "barostat.hpp"
#include "bound_potential.hpp"
#include "centroid_restraint.hpp"
#include "chiral_atom_restraint.hpp"
#include "chiral_bond_restraint.hpp"
#include "context.hpp"
#include "fanout_summed_potential.hpp"
#include "fixed_point.hpp"
#include "flat_bottom_bond.hpp"
#include "harmonic_angle.hpp"
#include "harmonic_angle_stable.hpp"
#include "harmonic_bond.hpp"
#include "langevin_integrator.hpp"
#include "local_md_utils.hpp"
#include "log_flat_bottom_bond.hpp"
#include "neighborlist.hpp"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_common.hpp"
#include "nonbonded_interaction_group.hpp"
#include "nonbonded_pair_list.hpp"
#include "nonbonded_precomputed.hpp"
#include "periodic_torsion.hpp"
#include "potential.hpp"
#include "rmsd_align.hpp"
#include "set_utils.hpp"
#include "summed_potential.hpp"
#include "verlet_integrator.hpp"

#include <iostream>

namespace py = pybind11;

// A utility to make sure that the coords and box shapes are correct
void verify_coords_and_box(
    const py::array_t<double, py::array::c_style> &coords, const py::array_t<double, py::array::c_style> &box) {
    size_t coord_dimensions = coords.ndim();
    if (coord_dimensions != 2) {
        throw std::runtime_error("coords dimensions must be 2");
    }
    if (coords.shape(coord_dimensions - 1) != 3) {
        throw std::runtime_error("coords must have a shape that is 3 dimensional");
    }
    if (box.ndim() != 2 || box.shape(0) != 3 || box.shape(1) != 3) {
        throw std::runtime_error("box must be 3x3");
    }
    auto box_data = box.data();
    for (int i = 0; i < box.size(); i++) {
        if (i == 0 || i == 4 || i == 8) {
            if (box_data[i] <= 0.0) {
                throw std::runtime_error("box must have positive values along diagonal");
            }
        } else if (box_data[i] != 0.0) {
            throw std::runtime_error("box must be ortholinear");
        }
    }
}

// accumulate_energies handles the combining of energies, summing them up deterministically
// and returning nan if there are overflows.
// The energies are collected in unsigned long long and overflows are incremented or decremented
// by the potentials. Currently only the negated NonbondedAllPairs decrements the overflows which
// are expected to exactly cancel out with other Nonbonded potentials if there are no clashes.
double accumulate_energies(int N, unsigned long long *fixed_u, int overflows) {
    double res = std::numeric_limits<double>::quiet_NaN();
    if (overflows == 0) {
        unsigned long long accum = 0;
        for (int i = 0; i < N; i++) {
            accum += fixed_u[i];
        }
        res = FIXED_TO_FLOAT<double>(accum);
    }
    return res;
}

template <typename RealType> void declare_neighborlist(py::module &m, const char *typestr) {

    using Class = timemachine::Neighborlist<RealType>;
    std::string pyclass_name = std::string("Neighborlist_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init([](int N) { return new timemachine::Neighborlist<RealType>(N); }), py::arg("N"))
        .def(
            "compute_block_bounds",
            [](timemachine::Neighborlist<RealType> &nblist,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box,
               const int block_size) -> py::tuple {
                if (block_size != 32) {
                    // The neighborlist kernel implementation assumes that block size is fixed to the CUDA warpSize
                    throw std::runtime_error("Block size must be 32.");
                }
                verify_coords_and_box(coords, box);
                int N = coords.shape()[0];
                int D = coords.shape()[1];
                int B = (N + block_size - 1) / block_size;

                py::array_t<double, py::array::c_style> py_bb_ctrs({B, D});
                py::array_t<double, py::array::c_style> py_bb_exts({B, D});

                nblist.compute_block_bounds_host(
                    N, coords.data(), box.data(), py_bb_ctrs.mutable_data(), py_bb_exts.mutable_data());

                return py::make_tuple(py_bb_ctrs, py_bb_exts);
            },
            py::arg("coords"),
            py::arg("box"),
            py::arg("block_size"))
        .def(
            "get_nblist",
            [](timemachine::Neighborlist<RealType> &nblist,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box,
               const double cutoff) -> std::vector<std::vector<int>> {
                int N = coords.shape()[0];
                verify_coords_and_box(coords, box);
                std::vector<std::vector<int>> ixn_list = nblist.get_nblist_host(N, coords.data(), box.data(), cutoff);

                return ixn_list;
            },
            py::arg("coords"),
            py::arg("box"),
            py::arg("cutoff"))
        .def(
            "set_row_idxs",
            [](timemachine::Neighborlist<RealType> &nblist,
               const py::array_t<unsigned int, py::array::c_style> &idxs_i) {
                std::vector<unsigned int> idxs(idxs_i.size());
                std::memcpy(idxs.data(), idxs_i.data(), idxs_i.size() * sizeof(unsigned int));
                nblist.set_row_idxs(idxs);
            },
            py::arg("idxs"))
        .def("reset_row_idxs", &timemachine::Neighborlist<RealType>::reset_row_idxs)
        .def("resize", &timemachine::Neighborlist<RealType>::resize, py::arg("size"));
}

void declare_context(py::module &m) {

    using Class = timemachine::Context;
    std::string pyclass_name = std::string("Context");
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<double, py::array::c_style> &x0,
                        const py::array_t<double, py::array::c_style> &v0,
                        const py::array_t<double, py::array::c_style> &box0,
                        std::shared_ptr<timemachine::Integrator> intg,
                        std::vector<std::shared_ptr<timemachine::BoundPotential>> bps,
                        std::optional<std::shared_ptr<timemachine::MonteCarloBarostat>> barostat) {
                int N = x0.shape()[0];
                int D = x0.shape()[1];
                verify_coords_and_box(x0, box0);
                if (N != v0.shape()[0]) {
                    throw std::runtime_error("v0 N != x0 N");
                }

                if (D != v0.shape()[1]) {
                    throw std::runtime_error("v0 D != x0 D");
                }

                return new timemachine::Context(
                    N, x0.data(), v0.data(), box0.data(), intg, bps, barostat.has_value() ? barostat.value() : nullptr);
            }),
            py::arg("x0"),
            py::arg("v0"),
            py::arg("box"),
            py::arg("integrator"),
            py::arg("bps"),
            py::arg("barostat") = py::none())
        .def(
            "step",
            &timemachine::Context::step,
            R"pbdoc(
        Take a single step.

        Note: Must call `initialize` before stepping and `finalize` after stepping to ensure the correct velocities and positions to be returned by `get_x_t()` and `get_v_t()`,.
        )pbdoc")
        .def("finalize", &timemachine::Context::finalize)
        .def("initialize", &timemachine::Context::initialize)
        .def(
            "multiple_steps",
            [](timemachine::Context &ctxt, const int n_steps, int store_x_interval) -> py::tuple {
                // (ytz): I hate C++
                int x_interval = (store_x_interval <= 0) ? n_steps : store_x_interval;
                std::array<std::vector<double>, 2> result = ctxt.multiple_steps(n_steps, x_interval);

                int N = ctxt.num_atoms();
                int D = 3;
                int F = result[0].size() / (N * D);
                py::array_t<double, py::array::c_style> out_x_buffer({F, N, D});
                std::memcpy(out_x_buffer.mutable_data(), result[0].data(), result[0].size() * sizeof(double));

                py::array_t<double, py::array::c_style> box_buffer({F, D, D});
                std::memcpy(box_buffer.mutable_data(), result[1].data(), result[1].size() * sizeof(double));

                return py::make_tuple(out_x_buffer, box_buffer);
            },
            py::arg("n_steps"),
            py::arg("store_x_interval") = 0,
            R"pbdoc(
        Take multiple steps.

        Frames are stored after having taken the number of steps specified by store_x_interval. E.g. if
        store_x_interval is 5, then on the 5th step the frame will be stored.

        Parameters
        ----------
        n_steps: int
            Number of steps

        store_x_interval: int
            How often we store the frames, store after every store_x_interval iterations. Setting to zero collects frames
            at the last step. Setting store_x_interval > n_steps will return no frames and skip runtime validation of box
            size.

        Returns
        -------
        2-tuple of coordinates, boxes
            F = floor(n_steps/store_x_interval).
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

        Raises
        ------
            RuntimeError:
                Box dimensions are invalid when a frame is collected

    )pbdoc")
        .def(
            "multiple_steps_local",
            [](timemachine::Context &ctxt,
               const int n_steps,
               const py::array_t<int, py::array::c_style> &local_idxs,
               const int burn_in,
               const int store_x_interval,
               const double radius,
               const double k,
               const int seed) -> py::tuple {
                if (n_steps <= 0) {
                    throw std::runtime_error("local steps must be at least one");
                }
                if (burn_in < 0) {
                    throw std::runtime_error("burn in steps must be greater or equal to zero");
                }
                timemachine::verify_local_md_parameters(radius, k);

                const int N = ctxt.num_atoms();
                const int x_interval = (store_x_interval <= 0) ? n_steps : store_x_interval;

                std::vector<int> vec_local_idxs(local_idxs.size());
                std::memcpy(vec_local_idxs.data(), local_idxs.data(), vec_local_idxs.size() * sizeof(int));
                timemachine::verify_atom_idxs(N, vec_local_idxs);

                std::array<std::vector<double>, 2> result =
                    ctxt.multiple_steps_local(n_steps, vec_local_idxs, burn_in, x_interval, radius, k, seed);
                const int D = 3;
                const int F = result[0].size() / (N * D);
                py::array_t<double, py::array::c_style> out_x_buffer({F, N, D});
                std::memcpy(out_x_buffer.mutable_data(), result[0].data(), result[0].size() * sizeof(double));

                py::array_t<double, py::array::c_style> box_buffer({F, D, D});
                std::memcpy(box_buffer.mutable_data(), result[1].data(), result[1].size() * sizeof(double));
                return py::make_tuple(out_x_buffer, box_buffer);
            },
            py::arg("n_steps"),
            py::arg("local_idxs"),
            py::arg("burn_in") = 500, // This is arbitrarily selected as a default, TODO make informed choice
            py::arg("store_x_interval") = 0,
            py::arg("radius") = 1.2,
            py::arg("k") = 10000.0,
            py::arg("seed") = 2022,
            R"pbdoc(
        Take multiple steps using particles selected based on the log probability using a random particle from the local_idxs,
        the random particle is frozen for all steps.

        Running a barostat and local MD at the same time are not currently supported. If a barostat is
        assigned to the context, the barostat won't run.

        Note: Running this multiple times with small number of steps (< 100) may result in a vacuum around the local idxs due to
        discretization error caused by switching on the restraint after a particle has moved beyond the radius.

        F = iterations / store_x_interval

        The first call to `multiple_steps_local` takes longer than subsequent calls, if setup_local_md has not been called previously,
        initializes potentials needed for local MD. The default local MD parameters are to freeze the reference and to
        use the temperature of the integrator, which must be a LangevinIntegrator.

        Parameters
        ----------
        n_steps: int
            Number of steps to run.

        local_idxs: np.array of int32
            The idxs that defines the atoms to use as the region(s) to run local MD. A random idx will be
            selected to be frozen and used as the center of the shell of particles to be simulated. The selected
            idx is constant across all steps.

        burn_in: int
            How many steps to run prior to storing frames. This is to handle the fact that the local simulation applies a
            restraint, and burn in helps equilibrate the local simulation. Running with small numbers of steps (< 100) is not recommended.

        store_x_interval: int
            How often we store the frames, store after every store_x_interval iterations. Setting to zero collects frames
            at the last step. Setting store_x_interval > n_steps will return no frames and skip runtime validation of box
            size.

        radius: float
            The radius in nanometers from the selected idx to simulate for local MD.

        k: float
            The flat bottom restraint K value to use for selection and restraint of atoms within the inner shell.

        seed: int
            The seed that is used to randomly select a particle to freeze and for the probabilistic selection of
            free particles. It is recommended to provide a new seed each time this function is called.

        Returns
        -------
        2-tuple of coordinates, boxes
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

        Raises
        ------
            RuntimeError:
                Box dimensions are invalid when a frame is collected

        Note: All boxes returned will be identical as local MD only runs under constant volume.
    )pbdoc")
        .def(
            "multiple_steps_local_selection",
            [](timemachine::Context &ctxt,
               const int n_steps,
               const int reference_idx,
               const py::array_t<int, py::array::c_style> &selection_idxs,
               const int burn_in,
               const int store_x_interval,
               const double radius,
               const double k) -> py::tuple {
                if (n_steps <= 0) {
                    throw std::runtime_error("local steps must be at least one");
                }
                if (burn_in < 0) {
                    throw std::runtime_error("burn in steps must be greater or equal to zero");
                }
                timemachine::verify_local_md_parameters(radius, k);

                const int N = ctxt.num_atoms();
                const int x_interval = (store_x_interval <= 0) ? n_steps : store_x_interval;

                if (reference_idx < 0 || reference_idx >= N) {
                    throw std::runtime_error("reference idx must be at least 0 and less than " + std::to_string(N));
                }
                std::vector<int> vec_selection_idxs(selection_idxs.size());
                std::memcpy(vec_selection_idxs.data(), selection_idxs.data(), vec_selection_idxs.size() * sizeof(int));
                timemachine::verify_atom_idxs(N, vec_selection_idxs);
                std::set<int> selection_set(vec_selection_idxs.begin(), vec_selection_idxs.end());
                if (selection_set.find(reference_idx) != selection_set.end()) {
                    throw std::runtime_error("reference idx must not be in selection idxs");
                }

                std::array<std::vector<double>, 2> result = ctxt.multiple_steps_local_selection(
                    n_steps, reference_idx, vec_selection_idxs, burn_in, x_interval, radius, k);
                const int D = 3;
                const int F = result[0].size() / (N * D);
                py::array_t<double, py::array::c_style> out_x_buffer({F, N, D});
                std::memcpy(out_x_buffer.mutable_data(), result[0].data(), result[0].size() * sizeof(double));

                py::array_t<double, py::array::c_style> box_buffer({F, D, D});
                std::memcpy(box_buffer.mutable_data(), result[1].data(), result[1].size() * sizeof(double));
                return py::make_tuple(out_x_buffer, box_buffer);
            },
            py::arg("n_steps"),
            py::arg("reference_idx"),
            py::arg("selection_idxs"),
            py::arg("burn_in") = 500, // This is arbitrarily selected as a default, TODO make informed choice
            py::arg("store_x_interval") = 0,
            py::arg("radius") = 1.2,
            py::arg("k") = 10000.0,
            R"pbdoc(
        Take multiple steps using a selection of free particles restrained to a reference particle. Useful for avoiding the bias
        introduced by switching on and off the restraint on different particles as is done with multiple_steps_local.

        Running a barostat and local MD at the same time are not currently supported. If a barostat is
        assigned to the context, the barostat won't run.

        Note: Running this multiple times with small number of steps (< 100) may result in a vacuum around the local idxs due to
        discretization error caused by switching on the restraint after a particle has moved beyond the radius.

        F = iterations / store_x_interval

        The first call to `multiple_steps_local_selection` takes longer than subsequent calls, if setup_local_md has not been called previously,
        initializes potentials needed for local MD. The default local MD parameters are to freeze the reference and to
        use the temperature of the integrator, which must be a LangevinIntegrator.

        Parameters
        ----------
        n_steps: int
            Number of steps to run.

        reference_idx: int
            Idx of particle to use as reference, will be frozen during steps.

        selection_idxs: np.array of int32
            The idxs of particles that should be free during local MD. Will be restrained to the particle specified by reference_idx particle using a
            flat bottom restraint which is defined by the radius and k values. Can be up to N - 1 particles, IE all particles except the reference_idx.

        burn_in: int
            How many steps to run prior to storing frames. This is to handle the fact that the local simulation applies a
            restraint, and burn in helps equilibrate the local simulation. Running with small numbers of steps (< 100) is not recommended.

        store_x_interval: int
            How often we store the frames, store after every store_x_interval iterations. Setting to zero collects frames
            at the last step. Setting store_x_interval > n_steps will return no frames and skip runtime validation of box
            size.

        radius: float
            The radius in nanometers from the reference idx to allow particles to be unrestrained in, afterwards apply a restraint to the reference particle.

        k: float
            The flat bottom restraint K value to use for restraint of atoms to the reference particle.

        Returns
        -------
        2-tuple of coordinates, boxes
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

        Raises
        ------
            RuntimeError:
                Box dimensions are invalid when a frame is collected

        Note: All boxes returned will be identical as local MD only runs under constant volume.
    )pbdoc")
        .def(
            "setup_local_md",
            &timemachine::Context::setup_local_md,
            py::arg("temperature"),
            py::arg("freeze_reference"),
            R"pbdoc(
        Configures the potential for local MD. This is automatically done when calling local MD methods,
        but can be done explicitly and with different parameters.

        Parameters
        ----------
        temperature: float
            Temperature in kelvin

        freeze_reference: bool
            Whether or not to freeze reference, otherwise applies restraint between frozen
            particles and the reference.

        Raises
        ------
            RuntimeError:
                Called a second time, can only be called once.
    )pbdoc")
        .def(
            "set_x_t",
            [](timemachine::Context &ctxt, const py::array_t<double, py::array::c_style> new_x_t) {
                if (new_x_t.shape()[0] != ctxt.num_atoms()) {
                    throw std::runtime_error("number of new coords disagree with current coords");
                }
                ctxt.set_x_t(new_x_t.data());
            },
            py::arg("coords"))
        .def(
            "set_v_t",
            [](timemachine::Context &ctxt, const py::array_t<double, py::array::c_style> new_v_t) {
                if (new_v_t.shape()[0] != ctxt.num_atoms()) {
                    throw std::runtime_error("number of new velocities disagree with current coords");
                }
                ctxt.set_v_t(new_v_t.data());
            },
            py::arg("velocities"))
        .def(
            "set_box",
            [](timemachine::Context &ctxt, const py::array_t<double, py::array::c_style> new_box_t) {
                if (new_box_t.size() != 9 || new_box_t.shape()[0] != 3) {
                    throw std::runtime_error("box must be 3x3");
                }
                ctxt.set_box(new_box_t.data());
            },
            py::arg("box"))
        .def(
            "get_x_t",
            [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
                unsigned int N = ctxt.num_atoms();
                unsigned int D = 3;
                py::array_t<double, py::array::c_style> buffer({N, D});
                ctxt.get_x_t(buffer.mutable_data());
                return buffer;
            })
        .def(
            "get_v_t",
            [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
                unsigned int N = ctxt.num_atoms();
                unsigned int D = 3;
                py::array_t<double, py::array::c_style> buffer({N, D});
                ctxt.get_v_t(buffer.mutable_data());
                return buffer;
            })
        .def("get_box", [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
            unsigned int D = 3;
            py::array_t<double, py::array::c_style> buffer({D, D});
            ctxt.get_box(buffer.mutable_data());
            return buffer;
        });
}

void declare_integrator(py::module &m) {

    using Class = timemachine::Integrator;
    std::string pyclass_name = std::string("Integrator");
    py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
}

void declare_langevin_integrator(py::module &m) {

    using Class = timemachine::LangevinIntegrator;
    std::string pyclass_name = std::string("LangevinIntegrator");
    py::class_<Class, std::shared_ptr<Class>, timemachine::Integrator>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<double, py::array::c_style> &masses,
                        double temperature,
                        double dt,
                        double friction,
                        int seed) {
                return new timemachine::LangevinIntegrator(
                    masses.size(), masses.data(), temperature, dt, friction, seed);
            }),
            py::arg("masses"),
            py::arg("temperature"),
            py::arg("dt"),
            py::arg("friction"),
            py::arg("seed"));
}

void declare_velocity_verlet_integrator(py::module &m) {

    using Class = timemachine::VelocityVerletIntegrator;
    std::string pyclass_name = std::string("VelocityVerletIntegrator");
    py::class_<Class, std::shared_ptr<Class>, timemachine::Integrator>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](double dt, const py::array_t<double, py::array::c_style> &cbs) {
                return new timemachine::VelocityVerletIntegrator(cbs.size(), dt, cbs.data());
            }),
            py::arg("dt"),
            py::arg("cbs"));
}

void declare_potential(py::module &m) {

    using Class = timemachine::Potential;
    std::string pyclass_name = std::string("Potential");
    py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            "execute",
            [](timemachine::Potential &pot,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &params,
               const py::array_t<double, py::array::c_style> &box) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                const long unsigned int P = params.size();
                verify_coords_and_box(coords, box);
                // initialize with fixed garbage values for debugging convenience (these should be overwritten by `execute_host`)
                std::vector<unsigned long long> du_dx(N * D, 9999);
                std::vector<unsigned long long> du_dp(P, 9999);
                std::vector<unsigned long long> u(1, 9999);
                std::vector<int> u_overflows(1, 9999);

                pot.execute_host(
                    N, P, coords.data(), params.data(), box.data(), &du_dx[0], &du_dp[0], &u[0], &u_overflows[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    // py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());

                py::array_t<double, py::array::c_style> py_du_dp(pshape);
                pot.du_dp_fixed_to_float(N, P, &du_dp[0], py_du_dp.mutable_data());

                double u_sum = accumulate_energies(1, &u[0], u_overflows[0]);

                return py::make_tuple(py_du_dx, py_du_dp, u_sum);
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("box"))
        .def(
            "execute_selective_batch",
            [](timemachine::Potential &pot,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &params,
               const py::array_t<double, py::array::c_style> &boxes,
               const bool compute_du_dx,
               const bool compute_du_dp,
               const bool compute_u) -> py::tuple {
                if (coords.ndim() != 3 && boxes.ndim() != 3) {
                    throw std::runtime_error("coords and boxes must have 3 dimensions");
                }
                if (coords.shape()[0] != boxes.shape()[0]) {
                    throw std::runtime_error("number of batches of coords and boxes don't match");
                }
                if (params.ndim() < 2) {
                    throw std::runtime_error("parameters must have at least 2 dimensions");
                }
                const long unsigned int coord_batches = coords.shape()[0];
                const long unsigned int N = coords.shape()[1];
                const long unsigned int D = coords.shape()[2];

                const long unsigned int param_batches = params.shape()[0];
                const long unsigned int P = params.size() / param_batches;

                const long unsigned int total_executions = coord_batches * param_batches;

                // initialize with fixed garbage values for debugging convenience (these should be overwritten by `execute_batch_host`)
                // Only initialize memory when needed, as buffers can be quite large
                std::vector<unsigned long long> du_dx;
                if (compute_du_dx) {
                    du_dx.assign(total_executions * N * D, 9999);
                }
                std::vector<unsigned long long> du_dp;
                if (compute_du_dp) {
                    du_dp.assign(total_executions * P, 9999);
                }
                std::vector<unsigned long long> u;
                std::vector<int> u_overflows;
                if (compute_u) {
                    u.assign(total_executions, 9999);
                    u_overflows.assign(total_executions, 9999);
                }

                pot.execute_batch_host(
                    coord_batches,
                    N,
                    param_batches,
                    P,
                    coords.data(),
                    params.data(),
                    boxes.data(),
                    compute_du_dx ? du_dx.data() : nullptr,
                    compute_du_dp ? du_dp.data() : nullptr,
                    compute_u ? u.data() : nullptr,
                    compute_u ? u_overflows.data() : nullptr);

                auto result = py::make_tuple(py::none(), py::none(), py::none());
                if (compute_du_dx) {
                    py::array_t<double, py::array::c_style> py_du_dx({coord_batches, param_batches, N, D});
                    for (unsigned int i = 0; i < du_dx.size(); i++) {
                        py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                    }
                    result[0] = py_du_dx;
                }

                if (compute_du_dp) {
                    std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());
                    // Remove the first dimension of the parameters shape to be consistent in ordering of return values
                    pshape.erase(pshape.begin());
                    // Append the new dimensions for the du_dps
                    unsigned long int shape[] = {coord_batches, param_batches};
                    pshape.insert(pshape.begin(), shape, shape + 2);

                    py::array_t<double, py::array::c_style> py_du_dp(pshape);
                    for (unsigned int i = 0; i < total_executions; i++) {
                        pot.du_dp_fixed_to_float(N, P, &du_dp[0] + (i * P), py_du_dp.mutable_data() + (i * P));
                    }
                    result[1] = py_du_dp;
                }

                if (compute_u) {
                    py::array_t<double, py::array::c_style> py_u({coord_batches, param_batches});

                    for (unsigned int i = 0; i < py_u.size(); i++) {
                        py_u.mutable_data()[i] = accumulate_energies(1, &u[0] + i, u_overflows[i]);
                    }
                    result[2] = py_u;
                }

                return result;
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("boxes"),
            py::arg("compute_du_dx"),
            py::arg("compute_du_dp"),
            py::arg("compute_u"),
            R"pbdoc(
        Execute the potential over a batch of coords and parameters. The total number of executions of the potential is
        num_coord_batches * num_param_batches.

        Note: This function allocates memory for all of the inputs on the GPU. This may lead to OOMs.

        Parameters
        ----------
        coords: NDArray
            A three dimensional array containing a batch of coordinates.

        params: NDArray
            A multi dimensional array containing a batch of parameters. First dimension
            determines the batch size, the rest of the array is passed to the potential as the
            parameters.

        boxes: NDArray
            A three dimensional array containing a batch of boxes.

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx.

        compute_du_dp: bool
            Indicates to compute du_dp, else returns None for du_dp.

        compute_u: bool
            Indicates to compute u, else returns None for u.


        Returns
        -------
        3-tuple of du_dx, du_dp, u
            coord_batch_size = coords.shape[0]
            param_batch_size = params.shape[0]
            du_dx has shape (coords_batch_size, param_batch_size, N, 3)
            du_dp has shape (coords_batch_size, param_batch_size, P)
            u has shape (coords_batch_size, param_batch_size)

    )pbdoc")
        .def(
            "execute_selective",
            [](timemachine::Potential &pot,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &params,
               const py::array_t<double, py::array::c_style> &box,
               bool compute_du_dx,
               bool compute_du_dp,
               bool compute_u) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                const long unsigned int P = params.size();
                verify_coords_and_box(coords, box);
                // initialize with fixed garbage values for debugging convenience (these should be overwritten by `execute_host`)
                std::vector<unsigned long long> du_dx(N * D, 9999);
                std::vector<unsigned long long> du_dp(P, 9999);

                std::vector<unsigned long long> u(1, 9999);
                std::vector<int> u_overflows(1, 9999);

                pot.execute_host(
                    N,
                    P,
                    coords.data(),
                    params.data(),
                    box.data(),
                    compute_du_dx ? &du_dx[0] : nullptr,
                    compute_du_dp ? &du_dp[0] : nullptr,
                    compute_u ? &u[0] : nullptr,
                    compute_u ? &u_overflows[0] : nullptr);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());

                py::array_t<double, py::array::c_style> py_du_dp(pshape);
                pot.du_dp_fixed_to_float(N, P, &du_dp[0], py_du_dp.mutable_data());

                double u_sum = accumulate_energies(1, &u[0], u_overflows[0]);

                auto result = py::make_tuple(py_du_dx, py_du_dp, u_sum);

                if (!compute_du_dx) {
                    result[0] = py::none();
                }
                if (!compute_du_dp) {
                    result[1] = py::none();
                }
                if (!compute_u) {
                    result[2] = py::none();
                }

                return result;
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("box"),
            py::arg("compute_du_dx"),
            py::arg("compute_du_dp"),
            py::arg("compute_u"))
        .def(
            "execute_du_dx",
            [](timemachine::Potential &pot,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &params,
               const py::array_t<double, py::array::c_style> &box) -> py::array_t<double, py::array::c_style> {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                const long unsigned int P = params.size();
                verify_coords_and_box(coords, box);

                std::vector<unsigned long long> du_dx(N * D);

                pot.execute_host_du_dx(N, P, coords.data(), params.data(), box.data(), &du_dx[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                return py_du_dx;
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("box"));
}

void declare_bound_potential(py::module &m) {

    using Class = timemachine::BoundPotential;
    std::string pyclass_name = std::string("BoundPotential");
    py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](std::shared_ptr<timemachine::Potential> potential,
                        const py::array_t<double, py::array::c_style> &params) {
                std::vector<int> pshape(params.shape(), params.shape() + params.ndim());

                return new timemachine::BoundPotential(potential, pshape, params.data());
            }),
            py::arg("potential"),
            py::arg("params"))
        .def("get_potential", [](const timemachine::BoundPotential &bp) { return bp.potential; })
        .def("size", &timemachine::BoundPotential::size)
        .def(
            "execute",
            [](timemachine::BoundPotential &bp,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                verify_coords_and_box(coords, box);
                std::vector<unsigned long long> du_dx(N * D, 9999);
                std::vector<unsigned long long> u(1, 9999);
                std::vector<int> u_overflows(1, 9999);

                bp.execute_host(N, coords.data(), box.data(), &du_dx[0], &u[0], &u_overflows[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                double u_sum = accumulate_energies(1, &u[0], u_overflows[0]);

                return py::make_tuple(py_du_dx, u_sum);
            },
            py::arg("coords"),
            py::arg("box"))
        .def(
            "execute_fixed",
            [](timemachine::BoundPotential &bp,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                verify_coords_and_box(coords, box);
                std::vector<unsigned long long> u(1, 9999);
                std::vector<int> u_overflows(1, 9999);

                bp.execute_host(N, coords.data(), box.data(), nullptr, &u[0], &u_overflows[0]);

                uint64_t u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));
                py::array_t<uint64_t, py::array::c_style> py_u(1);
                py_u.mutable_data()[0] = u_sum;
                return py::make_tuple(py_u, u_overflows[0]);
            },
            py::arg("coords"),
            py::arg("box"));
}

template <typename RealType> void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<RealType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &bond_idxs) {
                std::vector<int> vec_bond_idxs(bond_idxs.data(), bond_idxs.data() + bond_idxs.size());
                return new timemachine::HarmonicBond<RealType>(vec_bond_idxs);
            }),
            py::arg("bond_idxs"));
}

template <typename RealType> void declare_flat_bottom_bond(py::module &m, const char *typestr) {

    using Class = timemachine::FlatBottomBond<RealType>;
    std::string pyclass_name = std::string("FlatBottomBond_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &bond_idxs) {
                std::vector<int> vec_bond_idxs(bond_idxs.data(), bond_idxs.data() + bond_idxs.size());
                return new timemachine::FlatBottomBond<RealType>(vec_bond_idxs);
            }),
            py::arg("bond_idxs"));
}

template <typename RealType> void declare_log_flat_bottom_bond(py::module &m, const char *typestr) {

    using Class = timemachine::LogFlatBottomBond<RealType>;
    std::string pyclass_name = std::string("LogFlatBottomBond_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &bond_idxs, double beta) {
                std::vector<int> vec_bond_idxs(bond_idxs.data(), bond_idxs.data() + bond_idxs.size());
                return new timemachine::LogFlatBottomBond<RealType>(vec_bond_idxs, beta);
            }),
            py::arg("bond_idxs"),
            py::arg("beta"));
}

template <typename RealType> void declare_nonbonded_precomputed(py::module &m, const char *typestr) {

    using Class = timemachine::NonbondedPairListPrecomputed<RealType>;
    std::string pyclass_name = std::string("NonbondedPairListPrecomputed_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &pair_idxs, double beta, double cutoff) {
                std::vector<int> vec_pair_idxs(pair_idxs.data(), pair_idxs.data() + pair_idxs.size());
                return new timemachine::NonbondedPairListPrecomputed<RealType>(vec_pair_idxs, beta, cutoff);
            }),
            py::arg("pair_idxs"),
            py::arg("beta"),
            py::arg("cutoff"));
}

template <typename RealType> void declare_chiral_atom_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::ChiralAtomRestraint<RealType>;
    std::string pyclass_name = std::string("ChiralAtomRestraint_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &idxs) {
                std::vector<int> vec_idxs(idxs.data(), idxs.data() + idxs.size());
                return new timemachine::ChiralAtomRestraint<RealType>(vec_idxs);
            }),
            py::arg("idxs"),
            R"pbdoc(Please refer to timemachine.potentials.chiral_restraints for documentation on arguments)pbdoc");
}

template <typename RealType> void declare_chiral_bond_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::ChiralBondRestraint<RealType>;
    std::string pyclass_name = std::string("ChiralBondRestraint_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &idxs,
                        const py::array_t<int, py::array::c_style> &signs) {
                std::vector<int> vec_idxs(idxs.data(), idxs.data() + idxs.size());
                std::vector<int> vec_signs(signs.data(), signs.data() + signs.size());
                return new timemachine::ChiralBondRestraint<RealType>(vec_idxs, vec_signs);
            }),
            py::arg("idxs"),
            py::arg("signs"),
            R"pbdoc(Please refer to timemachine.potentials.chiral_restraints for documentation on arguments)pbdoc");
}

template <typename RealType> void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &angle_idxs) {
                std::vector<int> vec_angle_idxs(angle_idxs.size());
                std::memcpy(vec_angle_idxs.data(), angle_idxs.data(), vec_angle_idxs.size() * sizeof(int));
                return new timemachine::HarmonicAngle<RealType>(vec_angle_idxs);
            }),
            py::arg("angle_idxs"));
}

template <typename RealType> void declare_harmonic_angle_stable(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngleStable<RealType>;
    std::string pyclass_name = std::string("HarmonicAngleStable_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &angle_idxs) {
                std::vector<int> vec_angle_idxs(angle_idxs.size());
                std::memcpy(vec_angle_idxs.data(), angle_idxs.data(), vec_angle_idxs.size() * sizeof(int));
                return new timemachine::HarmonicAngleStable<RealType>(vec_angle_idxs);
            }),
            py::arg("angle_idxs"));
}

template <typename RealType> void declare_centroid_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::CentroidRestraint<RealType>;
    std::string pyclass_name = std::string("CentroidRestraint_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &group_a_idxs,
                        const py::array_t<int, py::array::c_style> &group_b_idxs,
                        double kb,
                        double b0) {
                std::vector<int> vec_group_a_idxs(group_a_idxs.size());
                std::memcpy(vec_group_a_idxs.data(), group_a_idxs.data(), vec_group_a_idxs.size() * sizeof(int));
                std::vector<int> vec_group_b_idxs(group_b_idxs.size());
                std::memcpy(vec_group_b_idxs.data(), group_b_idxs.data(), vec_group_b_idxs.size() * sizeof(int));

                return new timemachine::CentroidRestraint<RealType>(vec_group_a_idxs, vec_group_b_idxs, kb, b0);
            }),
            py::arg("group_a_idxs"),
            py::arg("group_b_idxs"),
            py::arg("kb"),
            py::arg("b0"));
}

template <typename RealType> void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<RealType>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &torsion_idxs) {
                std::vector<int> vec_torsion_idxs(torsion_idxs.size());
                std::memcpy(vec_torsion_idxs.data(), torsion_idxs.data(), vec_torsion_idxs.size() * sizeof(int));
                return new timemachine::PeriodicTorsion<RealType>(vec_torsion_idxs);
            }),
            py::arg("angle_idxs"));
}

template <typename RealType> void declare_nonbonded_all_pairs(py::module &m, const char *typestr) {

    using Class = timemachine::NonbondedAllPairs<RealType>;
    std::string pyclass_name = std::string("NonbondedAllPairs_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def("set_atom_idxs", &timemachine::NonbondedAllPairs<RealType>::set_atom_idxs, py::arg("atom_idxs"))
        .def("get_atom_idxs", &timemachine::NonbondedAllPairs<RealType>::get_atom_idxs)
        .def("get_num_atom_idxs", &timemachine::NonbondedAllPairs<RealType>::get_num_atom_idxs)
        .def(
            py::init([](const int N,
                        const double beta,
                        const double cutoff,
                        const std::optional<py::array_t<int, py::array::c_style>> &atom_idxs_i,
                        const bool disable_hilbert_sort,
                        const double nblist_padding) {
                std::optional<std::set<int>> unique_atom_idxs(std::nullopt);
                if (atom_idxs_i) {
                    std::vector<int> atom_idxs(atom_idxs_i->size());
                    std::memcpy(atom_idxs.data(), atom_idxs_i->data(), atom_idxs_i->size() * sizeof(int));
                    unique_atom_idxs.emplace(unique_idxs<int>(atom_idxs));
                }

                return new timemachine::NonbondedAllPairs<RealType>(
                    N, beta, cutoff, unique_atom_idxs, disable_hilbert_sort, nblist_padding);
            }),
            py::arg("num_atoms"),
            py::arg("beta"),
            py::arg("cutoff"),
            py::arg("atom_idxs_i") = py::none(),
            py::arg("disable_hilbert_sort") = false,
            py::arg("nblist_padding") = 0.1);
}

template <typename RealType> void declare_nonbonded_interaction_group(py::module &m, const char *typestr) {
    using Class = timemachine::NonbondedInteractionGroup<RealType>;
    std::string pyclass_name = std::string("NonbondedInteractionGroup_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            "set_atom_idxs",
            &timemachine::NonbondedInteractionGroup<RealType>::set_atom_idxs,
            py::arg("row_atom_idxs"),
            py::arg("col_atom_idxs"),
            R"pbdoc(
                    Set up the atom idxs for the NonbondedInteractionGroup.
                    The interaction is defined between two groups of atom idxs,
                    `row_atom_idxs` and `col_atom_idxs`. These should be a disjoint
                    list of idxs.

                    Parameters
                    ----------
                    row_atom_idxs: NDArray
                        First group of atoms in the interaction.

                    col_atom_idxs: NDArray
                        Second group of atoms in the interaction.

            )pbdoc")
        .def(
            py::init([](const int N,
                        const py::array_t<int, py::array::c_style> &row_atom_idxs_i,
                        const double beta,
                        const double cutoff,
                        std::optional<py::array_t<int, py::array::c_style>> &col_atom_idxs_i,
                        const bool disable_hilbert_sort,
                        const double nblist_padding) {
                std::vector<int> row_atom_idxs(row_atom_idxs_i.size());
                std::memcpy(row_atom_idxs.data(), row_atom_idxs_i.data(), row_atom_idxs_i.size() * sizeof(int));

                std::vector<int> col_atom_idxs;
                if (col_atom_idxs_i) {
                    col_atom_idxs.resize(col_atom_idxs_i->size());
                    std::memcpy(col_atom_idxs.data(), col_atom_idxs_i->data(), col_atom_idxs_i->size() * sizeof(int));
                } else {
                    std::set<int> unique_row_atom_idxs(unique_idxs(row_atom_idxs));
                    col_atom_idxs = get_indices_difference(N, unique_row_atom_idxs);
                }

                return new timemachine::NonbondedInteractionGroup<RealType>(
                    N, row_atom_idxs, col_atom_idxs, beta, cutoff, disable_hilbert_sort, nblist_padding);
            }),
            py::arg("num_atoms"),
            py::arg("row_atom_idxs_i"),
            py::arg("beta"),
            py::arg("cutoff"),
            py::arg("col_atom_idxs_i") = py::none(),
            py::arg("disable_hilbert_sort") = false,
            py::arg("nblist_padding") = 0.1,
            R"pbdoc(
                    Set up the NonbondedInteractionGroup.

                    Parameters
                    ----------
                    num_atoms: int
                        Number of atoms.

                    row_atom_idxs: NDArray
                        First group of atoms in the interaction.

                    beta: float

                    cutoff: float
                        Ignore all interactions beyond this distance in nm.

                    col_atom_idxs: Optional[NDArray]
                        Second group of atoms in the interaction. If not specified,
                        use all of the atoms not in the `row_atom_idxs`.

                    disable_hilbert_sort: bool
                        Set to True to disable the Hilbert sort.

                    nblist_padding: float
                        Margin for the neighborlist.

            )pbdoc");
}

template <typename RealType, bool Negated> void declare_nonbonded_pair_list(py::module &m, const char *typestr) {
    using Class = timemachine::NonbondedPairList<RealType, Negated>;
    std::string pyclass_name = std::string("NonbondedPairList_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &pair_idxs_i,
                        const py::array_t<double, py::array::c_style> &scales_i,
                        const double beta,
                        const double cutoff) {
                std::vector<int> pair_idxs(pair_idxs_i.size());
                std::memcpy(pair_idxs.data(), pair_idxs_i.data(), pair_idxs_i.size() * sizeof(int));

                std::vector<double> scales(scales_i.size());
                std::memcpy(scales.data(), scales_i.data(), scales_i.size() * sizeof(double));

                return new timemachine::NonbondedPairList<RealType, Negated>(pair_idxs, scales, beta, cutoff);
            }),
            py::arg("pair_idxs_i"),
            py::arg("scales_i"),
            py::arg("beta"),
            py::arg("cutoff"));
}

void declare_barostat(py::module &m) {

    using Class = timemachine::MonteCarloBarostat;
    std::string pyclass_name = std::string("MonteCarloBarostat");
    py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const int N,
                        const double pressure,
                        const double temperature,
                        std::vector<std::vector<int>> group_idxs,
                        const int frequency,
                        std::vector<std::shared_ptr<timemachine::BoundPotential>> bps,
                        const int seed) {
                return new timemachine::MonteCarloBarostat(N, pressure, temperature, group_idxs, frequency, bps, seed);
            }),
            py::arg("N"),
            py::arg("pressure"),
            py::arg("temperature"),
            py::arg("group_idxs"),
            py::arg("frequency"),
            py::arg("bps"),
            py::arg("seed"))
        .def("set_interval", &timemachine::MonteCarloBarostat::set_interval, py::arg("interval"))
        .def("get_interval", &timemachine::MonteCarloBarostat::get_interval)
        .def("set_pressure", &timemachine::MonteCarloBarostat::set_pressure, py::arg("pressure"));
}

void declare_summed_potential(py::module &m) {

    using Class = timemachine::SummedPotential;
    std::string pyclass_name = std::string("SummedPotential");
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init(
                [](std::vector<std::shared_ptr<timemachine::Potential>> potentials,
                   std::vector<int> params_sizes,
                   bool parallel) { return new timemachine::SummedPotential(potentials, params_sizes, parallel); }),

            py::arg("potentials"),
            py::arg("params_sizes"),
            py::arg("parallel") = true)
        .def("get_potentials", &timemachine::SummedPotential::get_potentials);
}

void declare_fanout_summed_potential(py::module &m) {

    using Class = timemachine::FanoutSummedPotential;
    std::string pyclass_name = std::string("FanoutSummedPotential");
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](std::vector<std::shared_ptr<timemachine::Potential>> potentials, bool parallel) {
                return new timemachine::FanoutSummedPotential(potentials, parallel);
            }),
            py::arg("potentials"),
            py::arg("parallel") = true)
        .def("get_potentials", &timemachine::FanoutSummedPotential::get_potentials);
}

const py::array_t<double, py::array::c_style>
py_rmsd_align(const py::array_t<double, py::array::c_style> &x1, const py::array_t<double, py::array::c_style> &x2) {

    int N1 = x1.shape()[0];
    int N2 = x2.shape()[0];

    int D1 = x1.shape()[1];
    int D2 = x2.shape()[1];

    if (N1 != N2) {
        throw std::runtime_error("N1 != N2");
    }

    if (D1 != 3) {
        throw std::runtime_error("D1 != 3");
    }

    if (D2 != 3) {
        throw std::runtime_error("D2 != 3");
    }

    py::array_t<double, py::array::c_style> py_x2_aligned({N1, D1});

    timemachine::rmsd_align_cpu(N1, x1.data(), x2.data(), py_x2_aligned.mutable_data());

    return py_x2_aligned;
}

void py_cuda_device_reset() { cudaDeviceReset(); }

PYBIND11_MODULE(custom_ops, m) {

    m.def(
        "cuda_device_reset",
        &py_cuda_device_reset,
        "Destroy all allocations and reset all state on the current device in the current process.");

    m.def("rmsd_align", &py_rmsd_align, "RMSD align two molecules", py::arg("x1"), py::arg("x2"));

    declare_barostat(m);

    declare_integrator(m);
    declare_langevin_integrator(m);
    declare_velocity_verlet_integrator(m);

    declare_potential(m);
    declare_bound_potential(m);
    declare_summed_potential(m);
    declare_fanout_summed_potential(m);

    declare_neighborlist<double>(m, "f64");
    declare_neighborlist<float>(m, "f32");

    declare_centroid_restraint<double>(m, "f64");
    declare_centroid_restraint<float>(m, "f32");

    declare_harmonic_bond<double>(m, "f64");
    declare_harmonic_bond<float>(m, "f32");

    declare_flat_bottom_bond<double>(m, "f64");
    declare_flat_bottom_bond<float>(m, "f32");

    declare_log_flat_bottom_bond<double>(m, "f64");
    declare_log_flat_bottom_bond<float>(m, "f32");

    declare_chiral_atom_restraint<double>(m, "f64");
    declare_chiral_atom_restraint<float>(m, "f32");

    declare_chiral_bond_restraint<double>(m, "f64");
    declare_chiral_bond_restraint<float>(m, "f32");

    declare_harmonic_angle<double>(m, "f64");
    declare_harmonic_angle<float>(m, "f32");

    declare_harmonic_angle_stable<double>(m, "f64");
    declare_harmonic_angle_stable<float>(m, "f32");

    declare_periodic_torsion<double>(m, "f64");
    declare_periodic_torsion<float>(m, "f32");

    declare_nonbonded_all_pairs<double>(m, "f64");
    declare_nonbonded_all_pairs<float>(m, "f32");

    declare_nonbonded_interaction_group<double>(m, "f64");
    declare_nonbonded_interaction_group<float>(m, "f32");

    declare_nonbonded_precomputed<double>(m, "f64");
    declare_nonbonded_precomputed<float>(m, "f32");

    declare_nonbonded_pair_list<double, false>(m, "f64");
    declare_nonbonded_pair_list<float, false>(m, "f32");

    declare_nonbonded_pair_list<double, true>(m, "f64_negated");
    declare_nonbonded_pair_list<float, true>(m, "f32_negated");

    declare_context(m);
}
