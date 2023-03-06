#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include "neighborlist.hpp"
#include "nonbonded_all_pairs.hpp"
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
                        timemachine::Integrator *intg,
                        std::vector<timemachine::BoundPotential *> bps,
                        std::optional<timemachine::MonteCarloBarostat *> barostat) {
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
                    N, x0.data(), v0.data(), box0.data(), intg, bps, barostat.has_value() ? barostat.value() : nullptr
                    // obs
                );
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
            How often we store the frames, stores after every store_x_interval steps

        Returns
        -------
        2-tuple of coordinates, boxes
            F = floor(n_steps/store_x_interval).
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

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
                    throw std::runtime_error("burn in steps must be greater than zero");
                }
                const int N = ctxt.num_atoms();
                const int x_interval = (store_x_interval <= 0) ? n_steps : store_x_interval;

                std::vector<int> vec_local_idxs(local_idxs.size());
                std::memcpy(vec_local_idxs.data(), local_idxs.data(), vec_local_idxs.size() * sizeof(int));
                if (vec_local_idxs.size() < 1) {
                    throw std::runtime_error("number of idxs must be at least 1");
                }
                if (vec_local_idxs.size() >= (long unsigned int)N) {
                    throw std::runtime_error("number of idxs must be less than N");
                }
                if (*std::max_element(vec_local_idxs.begin(), vec_local_idxs.end()) >= N) {
                    throw std::runtime_error("indices values must be less than N");
                }
                if (*std::min_element(vec_local_idxs.begin(), vec_local_idxs.end()) < 0) {
                    throw std::runtime_error("indices values must be greater than or equal to 0");
                }

                // Verify that local idxs are unique
                unique_idxs<int>(vec_local_idxs);
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
            at the last step.

        radius: float
            The radius in nanometers from the selected idx to simulate for local MD.

        k: float
            The flat bottom restraint K value to use for selection and restraint of atoms within the inner shell.

        seed: int
            The seed that is used to randomly select a particle to freeze and for the probabilistic selection of
            free particles. It is recommended to provide a unique seed at each call to `multiple_steps_local`.

        Returns
        -------
        2-tuple of coordinates, boxes
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

    )pbdoc")
        .def(
            "multiple_steps_U",
            [](timemachine::Context &ctxt, const int n_steps, int store_u_interval, int store_x_interval) -> py::tuple {
                int u_interval = (store_u_interval <= 0) ? n_steps : store_u_interval;
                int x_interval = (store_x_interval <= 0) ? n_steps : store_x_interval;

                std::array<std::vector<double>, 3> result = ctxt.multiple_steps_U(n_steps, u_interval, x_interval);

                int UF = n_steps / u_interval;

                py::array_t<double, py::array::c_style> out_u_buffer(UF);
                std::memcpy(out_u_buffer.mutable_data(), result[0].data(), result[0].size() * sizeof(double));

                int N = ctxt.num_atoms();
                int D = 3;
                int F = result[1].size() / (N * D);
                py::array_t<double, py::array::c_style> out_x_buffer({F, N, D});
                std::memcpy(out_x_buffer.mutable_data(), result[1].data(), result[1].size() * sizeof(double));

                py::array_t<double, py::array::c_style> box_buffer({F, D, D});
                std::memcpy(box_buffer.mutable_data(), result[2].data(), result[2].size() * sizeof(double));

                return py::make_tuple(out_u_buffer, out_x_buffer, box_buffer);
            },
            py::arg("n_steps"),
            py::arg("store_u_interval"),
            py::arg("store_x_interval"),
            R"pbdoc(
        Take multiple steps; store energies in addition to frames.

        Parameters
        ----------
        n_steps: int
            Number of steps to run.

        store_u_interval: int
            How often we store the energies, store after every store_u_interval steps

        store_x_interval: int
            How often we store the frames, store after every store_x_interval steps

        Returns
        -------
        3-tuple of energies, coordinates, boxes
            Energies have shape (F,)
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

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
                    throw std::runtime_error("number of new coords disagree with current coords");
                }
                ctxt.set_v_t(new_v_t.data());
            },
            py::arg("velocities"))
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
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
}

void declare_langevin_integrator(py::module &m) {

    using Class = timemachine::LangevinIntegrator;
    std::string pyclass_name = std::string("LangevinIntegrator");
    py::class_<Class, timemachine::Integrator>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
    py::class_<Class, timemachine::Integrator>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
                std::vector<unsigned long long> u(N, 9999);

                pot.execute_host(N, P, coords.data(), params.data(), box.data(), &du_dx[0], &du_dp[0], &u[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    // py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());

                py::array_t<double, py::array::c_style> py_du_dp(pshape);
                pot.du_dp_fixed_to_float(N, P, &du_dp[0], py_du_dp.mutable_data());

                unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

                return py::make_tuple(py_du_dx, py_du_dp, FIXED_TO_FLOAT<double>(u_sum));
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
                    du_dx.resize(total_executions * N * D, 9999);
                }
                std::vector<unsigned long long> du_dp;
                if (compute_du_dp) {
                    du_dp.resize(total_executions * P, 9999);
                }
                std::vector<unsigned long long> u;
                if (compute_u) {
                    // u vector is an array of unsigned long long that will be accumulated into a float, hence total_executions * N
                    u.resize(total_executions * N, 9999);
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
                    compute_u ? u.data() : nullptr);

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
                    for (unsigned int i = 0; i < total_executions; i++) {
                        unsigned long long u_sum =
                            std::accumulate(&u[0] + (i * N), &u[0] + ((i + 1) * N), decltype(u)::value_type(0));
                        py_u.mutable_data()[i] = FIXED_TO_FLOAT<double>(u_sum);
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

                std::vector<unsigned long long> u(N, 9999);

                pot.execute_host(
                    N,
                    P,
                    coords.data(),
                    params.data(),
                    box.data(),
                    compute_du_dx ? &du_dx[0] : nullptr,
                    compute_du_dp ? &du_dp[0] : nullptr,
                    compute_u ? &u[0] : nullptr);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());

                py::array_t<double, py::array::c_style> py_du_dp(pshape);
                pot.du_dp_fixed_to_float(N, P, &du_dp[0], py_du_dp.mutable_data());

                unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

                auto result = py::make_tuple(py_du_dx, py_du_dp, FIXED_TO_FLOAT<double>(u_sum));

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
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
                std::vector<unsigned long long> du_dx(N * D);
                std::vector<unsigned long long> u(N, 0);

                bp.execute_host(N, coords.data(), box.data(), &du_dx[0], &u[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

                return py::make_tuple(py_du_dx, FIXED_TO_FLOAT<double>(u_sum));
            },
            py::arg("coords"),
            py::arg("box"))
        .def(
            "execute_fixed",
            [](timemachine::BoundPotential &bp,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box) -> const py::array_t<uint64_t, py::array::c_style> {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                verify_coords_and_box(coords, box);
                // du_dx is computed, but not used
                std::vector<unsigned long long> du_dx(N * D);
                std::vector<unsigned long long> u(N, 0);

                bp.execute_host(N, coords.data(), box.data(), &du_dx[0], &u[0]);

                uint64_t u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));
                py::array_t<uint64_t, py::array::c_style> py_u(1);
                py_u.mutable_data()[0] = u_sum;
                return py_u;
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
        .def("set_nblist_padding", &timemachine::NonbondedAllPairs<RealType>::set_nblist_padding, py::arg("val"))
        .def("disable_hilbert_sort", &timemachine::NonbondedAllPairs<RealType>::disable_hilbert_sort)
        .def("set_atom_idxs", &timemachine::NonbondedAllPairs<RealType>::set_atom_idxs, py::arg("atom_idxs"))
        .def(
            py::init([](const int N,
                        const double beta,
                        const double cutoff,
                        const std::optional<py::array_t<int, py::array::c_style>> &atom_idxs_i) {
                std::optional<std::set<int>> unique_atom_idxs(std::nullopt);
                if (atom_idxs_i) {
                    std::vector<int> atom_idxs(atom_idxs_i->size());
                    std::memcpy(atom_idxs.data(), atom_idxs_i->data(), atom_idxs_i->size() * sizeof(int));
                    unique_atom_idxs.emplace(unique_idxs<int>(atom_idxs));
                }

                return new timemachine::NonbondedAllPairs<RealType>(N, beta, cutoff, unique_atom_idxs);
            }),
            py::arg("num_atoms"),
            py::arg("beta"),
            py::arg("cutoff"),
            py::arg("atom_idxs_i") = py::none());
}

template <typename RealType> void declare_nonbonded_interaction_group(py::module &m, const char *typestr) {
    using Class = timemachine::NonbondedInteractionGroup<RealType>;
    std::string pyclass_name = std::string("NonbondedInteractionGroup_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            "set_nblist_padding", &timemachine::NonbondedInteractionGroup<RealType>::set_nblist_padding, py::arg("val"))
        .def("disable_hilbert_sort", &timemachine::NonbondedInteractionGroup<RealType>::disable_hilbert_sort)
        .def(
            py::init([](const int N,
                        const py::array_t<int, py::array::c_style> &row_atom_idxs_i,
                        const double beta,
                        const double cutoff) {
                std::vector<int> row_atom_idxs(row_atom_idxs_i.size());
                std::memcpy(row_atom_idxs.data(), row_atom_idxs_i.data(), row_atom_idxs_i.size() * sizeof(int));
                std::set<int> unique_row_atom_idxs(unique_idxs(row_atom_idxs));

                return new timemachine::NonbondedInteractionGroup<RealType>(N, unique_row_atom_idxs, beta, cutoff);
            }),
            py::arg("num_atoms"),
            py::arg("row_atom_idxs_i"),
            py::arg("beta"),
            py::arg("cutoff"));
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
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const int N,
                        const double pressure,
                        const double temperature,
                        std::vector<std::vector<int>> group_idxs,
                        const int frequency,
                        std::vector<timemachine::BoundPotential *> bps,
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
                [](std::vector<std::shared_ptr<timemachine::Potential>> potentials, std::vector<int> params_sizes) {
                    return new timemachine::SummedPotential(potentials, params_sizes);
                }),

            py::arg("potentials"),
            py::arg("params_sizes"))
        .def("get_potentials", &timemachine::SummedPotential::get_potentials);
}

void declare_fanout_summed_potential(py::module &m) {

    using Class = timemachine::FanoutSummedPotential;
    std::string pyclass_name = std::string("FanoutSummedPotential");
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](std::vector<std::shared_ptr<timemachine::Potential>> potentials) {
                return new timemachine::FanoutSummedPotential(potentials);
            }),
            py::arg("potentials"))
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
