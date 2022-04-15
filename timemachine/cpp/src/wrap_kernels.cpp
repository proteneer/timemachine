#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>

#include "barostat.hpp"
#include "bound_potential.hpp"
#include "centroid_restraint.hpp"
#include "context.hpp"
#include "fanout_summed_potential.hpp"
#include "fixed_point.hpp"
#include "harmonic_angle.hpp"
#include "harmonic_bond.hpp"
#include "integrator.hpp"
#include "neighborlist.hpp"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_interaction_group.hpp"
#include "nonbonded_pair_list.hpp"
#include "periodic_torsion.hpp"
#include "potential.hpp"
#include "rmsd_align.hpp"
#include "summed_potential.hpp"

#include <iostream>

namespace py = pybind11;

template <typename RealType> void declare_neighborlist(py::module &m, const char *typestr) {

    using Class = timemachine::Neighborlist<RealType>;
    std::string pyclass_name = std::string("Neighborlist_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](int NC, std::optional<int> NR) {
                return new timemachine::Neighborlist<RealType>(NC, NR.has_value() ? NR.value() : 0);
            }),
            py::arg("NC"),
            py::arg("NR") = py::none())
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

                int N = coords.shape()[0];
                int D = coords.shape()[1];
                int B = (N + block_size - 1) / block_size;

                py::array_t<double, py::array::c_style> py_bb_ctrs({B, D});
                py::array_t<double, py::array::c_style> py_bb_exts({B, D});

                nblist.compute_block_bounds_host(
                    N, D, coords.data(), box.data(), py_bb_ctrs.mutable_data(), py_bb_exts.mutable_data());

                return py::make_tuple(py_bb_ctrs, py_bb_exts);
            })
        .def(
            "get_nblist",
            [](timemachine::Neighborlist<RealType> &nblist,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box,
               const double cutoff) -> std::vector<std::vector<int>> {
                int N = coords.shape()[0];

                std::vector<std::vector<int>> ixn_list = nblist.get_nblist_host(N, coords.data(), box.data(), cutoff);

                return ixn_list;
            })
        .def(
            "get_nblist_host_ligand",
            [](timemachine::Neighborlist<RealType> &nblist,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &row_coords,
               const py::array_t<double, py::array::c_style> &box,
               const double cutoff) -> std::vector<std::vector<int>> {
                const int N = coords.shape()[0];
                const int K = row_coords.shape()[0];

                std::vector<std::vector<int>> ixn_list =
                    nblist.get_nblist_host(N, K, coords.data(), row_coords.data(), box.data(), cutoff);

                return ixn_list;
            });
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
        .def("step", &timemachine::Context::step)
        .def(
            "multiple_steps",
            [](timemachine::Context &ctxt,
               const py::array_t<double, py::array::c_style> &lambda_schedule,
               int store_du_dl_interval,
               int store_x_interval) -> py::tuple {
                // (ytz): I hate C++
                std::vector<double> vec_lambda_schedule(lambda_schedule.size());
                std::memcpy(
                    vec_lambda_schedule.data(), lambda_schedule.data(), vec_lambda_schedule.size() * sizeof(double));

                int du_dl_interval = (store_du_dl_interval <= 0) ? lambda_schedule.size() : store_du_dl_interval;
                int x_interval = (store_x_interval <= 0) ? lambda_schedule.size() : store_x_interval;
                std::array<std::vector<double>, 3> result =
                    ctxt.multiple_steps(vec_lambda_schedule, du_dl_interval, x_interval);

                py::array_t<double, py::array::c_style> out_du_dl_buffer(result[0].size());
                std::memcpy(out_du_dl_buffer.mutable_data(), result[0].data(), result[0].size() * sizeof(double));

                int N = ctxt.num_atoms();
                int D = 3;
                int F = result[1].size() / (N * D);
                py::array_t<double, py::array::c_style> out_x_buffer({F, N, D});
                std::memcpy(out_x_buffer.mutable_data(), result[1].data(), result[1].size() * sizeof(double));

                py::array_t<double, py::array::c_style> box_buffer({F, D, D});
                std::memcpy(box_buffer.mutable_data(), result[2].data(), result[2].size() * sizeof(double));

                return py::make_tuple(out_du_dl_buffer, out_x_buffer, box_buffer);
            },
            py::arg("lambda_schedule"),
            py::arg("store_du_dl_interval") = 0,
            py::arg("store_x_interval") = 0)
        // .def("multiple_steps", &timemachine::Context::multiple_steps)
        .def(
            "multiple_steps_U",
            [](timemachine::Context &ctxt,
               const double lambda,
               const int n_steps,
               const py::array_t<double, py::array::c_style> &lambda_windows,
               int store_u_interval,
               int store_x_interval) -> py::tuple {
                std::vector<double> vec_lambda_windows(lambda_windows.size());
                std::memcpy(
                    vec_lambda_windows.data(), lambda_windows.data(), vec_lambda_windows.size() * sizeof(double));

                int u_interval = (store_u_interval <= 0) ? n_steps : store_u_interval;
                int x_interval = (store_x_interval <= 0) ? n_steps : store_x_interval;

                std::array<std::vector<double>, 3> result =
                    ctxt.multiple_steps_U(lambda, n_steps, vec_lambda_windows, u_interval, x_interval);

                int UW = lambda_windows.size();
                int UF = result[0].size() > 0 ? result[0].size() / UW : 0;

                py::array_t<double, py::array::c_style> out_u_buffer({UF, UW});
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
            py::arg("lamb"),
            py::arg("n_steps"),
            py::arg("lambda_windows"),
            py::arg("store_u_interval"),
            py::arg("store_x_interval"),
            R"pbdoc(
        Compute energies across multiple lambda windows while simulating
        at a single fixed lambda window.

        Let lambda_windows have shape [K].
        F = ceil(n_steps/store_u_interval).

        The returned U matrix has F rows and K columns.

        Parameters
        ----------
        lambda: float
            Lambda window we run the simulation at

        n_steps: int
            Number of steps to run.

        lambda_windows: np.array, of shape K
            Lambda values to evaluate energies at

        store_u_interval: int
            How often we store the energies

        store_x_interval: int
            How often we store the frames

        Returns
        -------
        3-tuple of energies, coordinates, boxes
            Energies have shape (F, K)
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

    )pbdoc")
        .def(
            "set_x_t",
            [](timemachine::Context &ctxt, const py::array_t<double, py::array::c_style> new_x_t) {
                ctxt.set_x_t(new_x_t.data());
            })
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
        .def(
            "get_box",
            [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
                unsigned int D = 3;
                py::array_t<double, py::array::c_style> buffer({D, D});
                ctxt.get_box(buffer.mutable_data());
                return buffer;
            })
        .def("_get_du_dx_t_minus_1", [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
            PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "_get_du_dx_t_minus_1() should only be used for testing. It will be removed in a future release.",
                1);
            unsigned int N = ctxt.num_atoms();
            unsigned int D = 3;
            std::vector<unsigned long long> du_dx(N * D);
            ctxt.get_du_dx_t_minus_1(&du_dx[0]);
            py::array_t<double, py::array::c_style> py_du_dx({N, D});
            for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
            }
            return py_du_dx;
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
            py::init([](double dt,
                        double ca,
                        const py::array_t<double, py::array::c_style> &cbs,
                        const py::array_t<double, py::array::c_style> &ccs,
                        int seed) {
                return new timemachine::LangevinIntegrator(cbs.size(), dt, ca, cbs.data(), ccs.data(), seed);
            }),
            py::arg("dt"),
            py::arg("ca"),
            py::arg("cbs"),
            py::arg("ccs"),
            py::arg("seed"));
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
               const py::array_t<double, py::array::c_style> &box,
               double lambda) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                const long unsigned int P = params.size();

                // initialize with fixed garbage values for debugging convenience (these should be overwritten by `execute_host`)
                std::vector<unsigned long long> du_dx(N * D, 9999);
                std::vector<unsigned long long> du_dp(P, 9999);
                std::vector<unsigned long long> du_dl(N, 9999);
                std::vector<unsigned long long> u(N, 9999);

                pot.execute_host(
                    N, P, coords.data(), params.data(), box.data(), lambda, &du_dx[0], &du_dp[0], &du_dl[0], &u[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    // py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());

                py::array_t<double, py::array::c_style> py_du_dp(pshape);
                pot.du_dp_fixed_to_float(N, P, &du_dp[0], py_du_dp.mutable_data());

                unsigned long long du_dl_sum =
                    std::accumulate(du_dl.begin(), du_dl.end(), decltype(du_dl)::value_type(0));
                unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

                return py::make_tuple(
                    py_du_dx, py_du_dp, FIXED_TO_FLOAT<double>(du_dl_sum), FIXED_TO_FLOAT<double>(u_sum));
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("box"),
            py::arg("lam"))
        .def(
            "execute_selective",
            [](timemachine::Potential &pot,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &params,
               const py::array_t<double, py::array::c_style> &box,
               double lambda,
               bool compute_du_dx,
               bool compute_du_dp,
               bool compute_du_dl,
               bool compute_u) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                const long unsigned int P = params.size();

                std::vector<unsigned long long> du_dx(N * D);
                std::vector<unsigned long long> du_dp(P);

                std::vector<unsigned long long> du_dl(N, 0);
                std::vector<unsigned long long> u(N, 0);

                pot.execute_host(
                    N,
                    P,
                    coords.data(),
                    params.data(),
                    box.data(),
                    lambda,
                    compute_du_dx ? &du_dx[0] : nullptr,
                    compute_du_dp ? &du_dp[0] : nullptr,
                    compute_du_dl ? &du_dl[0] : nullptr,
                    compute_u ? &u[0] : nullptr);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                std::vector<ssize_t> pshape(params.shape(), params.shape() + params.ndim());

                py::array_t<double, py::array::c_style> py_du_dp(pshape);
                pot.du_dp_fixed_to_float(N, P, &du_dp[0], py_du_dp.mutable_data());

                unsigned long long du_dl_sum =
                    std::accumulate(du_dl.begin(), du_dl.end(), decltype(du_dl)::value_type(0));
                unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

                auto result = py::make_tuple(
                    py_du_dx, py_du_dp, FIXED_TO_FLOAT<double>(du_dl_sum), FIXED_TO_FLOAT<double>(u_sum));

                if (!compute_du_dx) {
                    result[0] = py::none();
                }
                if (!compute_du_dp) {
                    result[1] = py::none();
                }
                if (!compute_du_dl) {
                    result[2] = py::none();
                }
                if (!compute_u) {
                    result[3] = py::none();
                }

                return result;
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("box"),
            py::arg("lam"),
            py::arg("compute_du_dx"),
            py::arg("compute_du_dp"),
            py::arg("compute_du_dl"),
            py::arg("compute_u"))
        .def(
            "execute_du_dx",
            [](timemachine::Potential &pot,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &params,
               const py::array_t<double, py::array::c_style> &box,
               double lambda) -> py::array_t<double, py::array::c_style> {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];
                const long unsigned int P = params.size();

                std::vector<unsigned long long> du_dx(N * D);

                pot.execute_host_du_dx(N, P, coords.data(), params.data(), box.data(), lambda, &du_dx[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                return py_du_dx;
            },
            py::arg("coords"),
            py::arg("params"),
            py::arg("box"),
            py::arg("lam"));
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
               const py::array_t<double, py::array::c_style> &box,
               double lambda) -> py::tuple {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];

                std::vector<unsigned long long> du_dx(N * D);
                std::vector<unsigned long long> du_dl(N, 0);
                std::vector<unsigned long long> u(N, 0);

                bp.execute_host(N, coords.data(), box.data(), lambda, &du_dx[0], &du_dl[0], &u[0]);

                py::array_t<double, py::array::c_style> py_du_dx({N, D});
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                    py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
                }

                unsigned long long du_dl_sum =
                    std::accumulate(du_dl.begin(), du_dl.end(), decltype(du_dl)::value_type(0));
                unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

                return py::make_tuple(py_du_dx, FIXED_TO_FLOAT<double>(du_dl_sum), FIXED_TO_FLOAT<double>(u_sum));
            },
            py::arg("coords"),
            py::arg("box"),
            py::arg("lam"))
        .def(
            "execute_fixed",
            [](timemachine::BoundPotential &bp,
               const py::array_t<double, py::array::c_style> &coords,
               const py::array_t<double, py::array::c_style> &box,
               double lambda) -> const py::array_t<uint64_t, py::array::c_style> {
                const long unsigned int N = coords.shape()[0];
                const long unsigned int D = coords.shape()[1];

                // du_dx and du_dl are computed, but not used
                std::vector<unsigned long long> du_dx(N * D);
                std::vector<unsigned long long> du_dl(N, 0);
                std::vector<unsigned long long> u(N, 0);

                bp.execute_host(N, coords.data(), box.data(), lambda, &du_dx[0], &du_dl[0], &u[0]);

                uint64_t u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));
                py::array_t<uint64_t, py::array::c_style> py_u(1);
                py_u.mutable_data()[0] = u_sum;
                return py_u;
            },
            py::arg("coords"),
            py::arg("box"),
            py::arg("lam"));
}

template <typename RealType> void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<RealType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &bond_idxs,
                        std::optional<py::array_t<int, py::array::c_style>> lamb_mult,
                        std::optional<py::array_t<int, py::array::c_style>> lamb_offset) {
                std::vector<int> vec_bond_idxs(bond_idxs.data(), bond_idxs.data() + bond_idxs.size());
                std::vector<int> vec_lamb_mult;
                std::vector<int> vec_lamb_offset;
                if (lamb_mult.has_value()) {
                    vec_lamb_mult.assign(lamb_mult.value().data(), lamb_mult.value().data() + lamb_mult.value().size());
                }
                if (lamb_offset.has_value()) {
                    vec_lamb_offset.assign(
                        lamb_offset.value().data(), lamb_offset.value().data() + lamb_offset.value().size());
                }
                return new timemachine::HarmonicBond<RealType>(vec_bond_idxs, vec_lamb_mult, vec_lamb_offset);
            }),
            py::arg("bond_idxs"),
            py::arg("lamb_mult") = py::none(),
            py::arg("lamb_offset") = py::none());
}

template <typename RealType> void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &angle_idxs,
                        std::optional<py::array_t<int, py::array::c_style>> lamb_mult,
                        std::optional<py::array_t<int, py::array::c_style>> lamb_offset) {
                std::vector<int> vec_angle_idxs(angle_idxs.size());
                std::memcpy(vec_angle_idxs.data(), angle_idxs.data(), vec_angle_idxs.size() * sizeof(int));
                std::vector<int> vec_lamb_mult;
                std::vector<int> vec_lamb_offset;
                if (lamb_mult.has_value()) {
                    vec_lamb_mult.assign(lamb_mult.value().data(), lamb_mult.value().data() + lamb_mult.value().size());
                }
                if (lamb_offset.has_value()) {
                    vec_lamb_offset.assign(
                        lamb_offset.value().data(), lamb_offset.value().data() + lamb_offset.value().size());
                }
                return new timemachine::HarmonicAngle<RealType>(vec_angle_idxs, vec_lamb_mult, vec_lamb_offset);
            }),
            py::arg("angle_idxs"),
            py::arg("lamb_mult") = py::none(),
            py::arg("lamb_offset") = py::none());
}

template <typename RealType> void declare_centroid_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::CentroidRestraint<RealType>;
    std::string pyclass_name = std::string("CentroidRestraint_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init([](const py::array_t<int, py::array::c_style> &group_a_idxs,
                         const py::array_t<int, py::array::c_style> &group_b_idxs,
                         double kb,
                         double b0) {
            std::vector<int> vec_group_a_idxs(group_a_idxs.size());
            std::memcpy(vec_group_a_idxs.data(), group_a_idxs.data(), vec_group_a_idxs.size() * sizeof(int));
            std::vector<int> vec_group_b_idxs(group_b_idxs.size());
            std::memcpy(vec_group_b_idxs.data(), group_b_idxs.data(), vec_group_b_idxs.size() * sizeof(int));

            return new timemachine::CentroidRestraint<RealType>(vec_group_a_idxs, vec_group_b_idxs, kb, b0);
        }));
}

template <typename RealType> void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<RealType>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const py::array_t<int, py::array::c_style> &torsion_idxs,
                        std::optional<py::array_t<int, py::array::c_style>> lamb_mult,
                        std::optional<py::array_t<int, py::array::c_style>> lamb_offset) {
                std::vector<int> vec_torsion_idxs(torsion_idxs.size());
                std::memcpy(vec_torsion_idxs.data(), torsion_idxs.data(), vec_torsion_idxs.size() * sizeof(int));
                std::vector<int> vec_lamb_mult;
                std::vector<int> vec_lamb_offset;
                if (lamb_mult.has_value()) {
                    vec_lamb_mult.assign(lamb_mult.value().data(), lamb_mult.value().data() + lamb_mult.value().size());
                }
                if (lamb_offset.has_value()) {
                    vec_lamb_offset.assign(
                        lamb_offset.value().data(), lamb_offset.value().data() + lamb_offset.value().size());
                }
                return new timemachine::PeriodicTorsion<RealType>(vec_torsion_idxs, vec_lamb_mult, vec_lamb_offset);
            }),
            py::arg("angle_idxs"),
            py::arg("lamb_mult") = py::none(),
            py::arg("lamb_offset") = py::none());
}

// stackoverflow
std::string dirname(const std::string &fname) {
    size_t pos = fname.find_last_of("\\/");
    return (std::string::npos == pos) ? "" : fname.substr(0, pos);
}

std::set<int> unique_idxs(const std::vector<int> &idxs) {
    std::set<int> unique_idxs(idxs.begin(), idxs.end());
    if (unique_idxs.size() < idxs.size()) {
        throw std::runtime_error("atom indices must be unique");
    }
    return unique_idxs;
}

template <typename RealType, bool Interpolated> void declare_nonbonded_all_pairs(py::module &m, const char *typestr) {

    using Class = timemachine::NonbondedAllPairs<RealType, Interpolated>;
    std::string pyclass_name = std::string("NonbondedAllPairs_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def("set_nblist_padding", &timemachine::NonbondedAllPairs<RealType, Interpolated>::set_nblist_padding)
        .def("disable_hilbert_sort", &timemachine::NonbondedAllPairs<RealType, Interpolated>::disable_hilbert_sort)
        .def(
            py::init([](const std::string &kernel_dir,
                        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i,
                        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i,
                        const double beta,
                        const double cutoff,
                        const std::optional<py::array_t<int, py::array::c_style>> &atom_idxs_i,
                        const std::string &transform_lambda_charge = "lambda",
                        const std::string &transform_lambda_sigma = "lambda",
                        const std::string &transform_lambda_epsilon = "lambda",
                        const std::string &transform_lambda_w = "lambda") {
                std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
                std::memcpy(
                    lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size() * sizeof(int));

                std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
                std::memcpy(
                    lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size() * sizeof(int));

                std::optional<std::set<int>> unique_atom_idxs(std::nullopt);
                if (atom_idxs_i) {
                    std::vector<int> atom_idxs(atom_idxs_i->size());
                    std::memcpy(atom_idxs.data(), atom_idxs_i->data(), atom_idxs_i->size() * sizeof(int));
                    unique_atom_idxs.emplace(unique_idxs(atom_idxs));
                }

                std::string src_path = kernel_dir + "/k_lambda_transformer_jit.cuh";
                std::ifstream t(src_path);
                std::string source_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
                source_str = std::regex_replace(source_str, std::regex("KERNEL_DIR"), kernel_dir);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_CHARGE"), transform_lambda_charge);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_SIGMA"), transform_lambda_sigma);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_EPSILON"), transform_lambda_epsilon);
                source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_W"), transform_lambda_w);

                return new timemachine::NonbondedAllPairs<RealType, Interpolated>(
                    lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, unique_atom_idxs, source_str);
            }),
            py::arg("kernel_dir"),
            py::arg("lambda_plane_idxs_i"),
            py::arg("lambda_offset_idxs_i"),
            py::arg("beta"),
            py::arg("cutoff"),
            py::arg("atom_idxs_i") = py::none(),
            py::arg("transform_lambda_charge") = "lambda",
            py::arg("transform_lambda_sigma") = "lambda",
            py::arg("transform_lambda_epsilon") = "lambda",
            py::arg("transform_lambda_w") = "lambda");
}

template <typename RealType, bool Interpolated>
void declare_nonbonded_interaction_group(py::module &m, const char *typestr) {
    using Class = timemachine::NonbondedInteractionGroup<RealType, Interpolated>;
    std::string pyclass_name = std::string("NonbondedInteractionGroup_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def("set_nblist_padding", &timemachine::NonbondedInteractionGroup<RealType, Interpolated>::set_nblist_padding)
        .def(
            "disable_hilbert_sort",
            &timemachine::NonbondedInteractionGroup<RealType, Interpolated>::disable_hilbert_sort)
        .def(
            py::init([](const std::string &kernel_dir,
                        const py::array_t<int, py::array::c_style> &row_atom_idxs_i,
                        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i,
                        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i,
                        const double beta,
                        const double cutoff,
                        const std::string &transform_lambda_charge = "lambda",
                        const std::string &transform_lambda_sigma = "lambda",
                        const std::string &transform_lambda_epsilon = "lambda",
                        const std::string &transform_lambda_w = "lambda") {
                std::vector<int> row_atom_idxs(row_atom_idxs_i.size());
                std::memcpy(row_atom_idxs.data(), row_atom_idxs_i.data(), row_atom_idxs_i.size() * sizeof(int));
                std::set<int> unique_row_atom_idxs(unique_idxs(row_atom_idxs));

                std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
                std::memcpy(
                    lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size() * sizeof(int));

                std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
                std::memcpy(
                    lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size() * sizeof(int));

                std::string src_path = kernel_dir + "/k_lambda_transformer_jit.cuh";
                std::ifstream t(src_path);
                std::string source_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
                source_str = std::regex_replace(source_str, std::regex("KERNEL_DIR"), kernel_dir);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_CHARGE"), transform_lambda_charge);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_SIGMA"), transform_lambda_sigma);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_EPSILON"), transform_lambda_epsilon);
                source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_W"), transform_lambda_w);

                return new timemachine::NonbondedInteractionGroup<RealType, Interpolated>(
                    unique_row_atom_idxs, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, source_str);
            }),
            py::arg("kernel_dir"),
            py::arg("row_atom_idxs_i"),
            py::arg("lambda_plane_idxs_i"),
            py::arg("lambda_offset_idxs_i"),
            py::arg("beta"),
            py::arg("cutoff"),
            py::arg("transform_lambda_charge") = "lambda",
            py::arg("transform_lambda_sigma") = "lambda",
            py::arg("transform_lambda_epsilon") = "lambda",
            py::arg("transform_lambda_w") = "lambda");
}

template <typename RealType, bool Negated, bool Interpolated>
void declare_nonbonded_pair_list(py::module &m, const char *typestr) {
    using Class = timemachine::NonbondedPairList<RealType, Negated, Interpolated>;
    std::string pyclass_name = std::string("NonbondedPairList_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(
            py::init([](const std::string &kernel_dir,
                        const py::array_t<int, py::array::c_style> &pair_idxs_i,
                        const py::array_t<double, py::array::c_style> &scales_i,
                        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i,
                        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i,
                        const double beta,
                        const double cutoff,
                        const std::string &transform_lambda_charge = "lambda",
                        const std::string &transform_lambda_sigma = "lambda",
                        const std::string &transform_lambda_epsilon = "lambda",
                        const std::string &transform_lambda_w = "lambda") {
                std::vector<int> pair_idxs(pair_idxs_i.size());
                std::memcpy(pair_idxs.data(), pair_idxs_i.data(), pair_idxs_i.size() * sizeof(int));

                std::vector<double> scales(scales_i.size());
                std::memcpy(scales.data(), scales_i.data(), scales_i.size() * sizeof(double));

                std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
                std::memcpy(
                    lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size() * sizeof(int));

                std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
                std::memcpy(
                    lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size() * sizeof(int));

                std::string src_path = kernel_dir + "/k_lambda_transformer_jit.cuh";
                std::ifstream t(src_path);
                std::string source_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
                source_str = std::regex_replace(source_str, std::regex("KERNEL_DIR"), kernel_dir);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_CHARGE"), transform_lambda_charge);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_SIGMA"), transform_lambda_sigma);
                source_str =
                    std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_EPSILON"), transform_lambda_epsilon);
                source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_W"), transform_lambda_w);

                return new timemachine::NonbondedPairList<RealType, Negated, Interpolated>(
                    pair_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, source_str);
            }),
            py::arg("kernel_dir"),
            py::arg("pair_idxs_i"),
            py::arg("scales_i"),
            py::arg("lambda_plane_idxs_i"),
            py::arg("lambda_offset_idxs_i"),
            py::arg("beta"),
            py::arg("cutoff"),
            py::arg("transform_lambda_charge") = "lambda",
            py::arg("transform_lambda_sigma") = "lambda",
            py::arg("transform_lambda_epsilon") = "lambda",
            py::arg("transform_lambda_w") = "lambda");
}

void declare_barostat(py::module &m) {

    using Class = timemachine::MonteCarloBarostat;
    std::string pyclass_name = std::string("MonteCarloBarostat");
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init([](const int N,
                         const double pressure,
                         const double temperature,
                         std::vector<std::vector<int>> group_idxs,
                         const int frequency,
                         std::vector<timemachine::BoundPotential *> bps,
                         const int seed) {
            return new timemachine::MonteCarloBarostat(N, pressure, temperature, group_idxs, frequency, bps, seed);
        }))
        .def("set_interval", &timemachine::MonteCarloBarostat::set_interval)
        .def("get_interval", &timemachine::MonteCarloBarostat::get_interval)
        .def("set_pressure", &timemachine::MonteCarloBarostat::set_pressure);
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

    m.def("rmsd_align", &py_rmsd_align, "RMSD align two molecules");

    declare_barostat(m);

    declare_integrator(m);
    declare_langevin_integrator(m);

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

    declare_harmonic_angle<double>(m, "f64");
    declare_harmonic_angle<float>(m, "f32");

    declare_periodic_torsion<double>(m, "f64");
    declare_periodic_torsion<float>(m, "f32");

    declare_nonbonded_all_pairs<double, true>(m, "f64_interpolated");
    declare_nonbonded_all_pairs<float, true>(m, "f32_interpolated");

    declare_nonbonded_all_pairs<double, false>(m, "f64");
    declare_nonbonded_all_pairs<float, false>(m, "f32");

    declare_nonbonded_interaction_group<double, true>(m, "f64_interpolated");
    declare_nonbonded_interaction_group<float, true>(m, "f32_interpolated");

    declare_nonbonded_interaction_group<double, false>(m, "f64");
    declare_nonbonded_interaction_group<float, false>(m, "f32");

    declare_nonbonded_pair_list<double, false, false>(m, "f64");
    declare_nonbonded_pair_list<float, false, false>(m, "f32");

    declare_nonbonded_pair_list<double, false, true>(m, "f64_interpolated");
    declare_nonbonded_pair_list<float, false, true>(m, "f32_interpolated");

    declare_nonbonded_pair_list<double, true, false>(m, "f64_negated");
    declare_nonbonded_pair_list<float, true, false>(m, "f32_negated");

    declare_nonbonded_pair_list<double, true, true>(m, "f64_negated_interpolated");
    declare_nonbonded_pair_list<float, true, true>(m, "f32_negated_interpolated");

    declare_context(m);
}
