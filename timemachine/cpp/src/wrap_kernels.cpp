#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <numeric>
#include <regex>

#include "context.hpp"
#include "potential.hpp"
#include "bound_potential.hpp"
#include "harmonic_bond.hpp"
#include "harmonic_angle.hpp"
// #include "lambda_potential.hpp"
// #include "interpolated_potential.hpp"
// #include "restraint.hpp"
// #include "inertial_restraint.hpp"
#include "centroid_restraint.hpp"
#include "rmsd_restraint.hpp"
#include "periodic_torsion.hpp"
#include "nonbonded.hpp"
// #include "lennard_jones.hpp"
// #include "electrostatics.hpp"
// #include "gbsa.hpp"
#include "fixed_point.hpp"
#include "integrator.hpp"
// #include "observable.hpp"
#include "neighborlist.hpp"
// #include "shape.hpp"


#include <iostream>

namespace py = pybind11;


template <typename RealType>
void declare_neighborlist(py::module &m, const char *typestr) {

    using Class = timemachine::Neighborlist<RealType>;
    std::string pyclass_name = std::string("Neighborlist_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](int N) {
        return new timemachine::Neighborlist<RealType>(N);
    }))
    .def("compute_block_bounds", [](
        timemachine::Neighborlist<RealType> &nblist,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &box,
        int block_size) -> py::tuple {

        if(block_size != 32) {
            throw std::runtime_error("Block size must be 32.");
        }

        int N = coords.shape()[0];
        int D = coords.shape()[1];
        int B = (N + block_size - 1)/block_size;

        py::array_t<double, py::array::c_style> py_bb_ctrs({B, D});
        py::array_t<double, py::array::c_style> py_bb_exts({B, D});

        nblist.compute_block_bounds_host(
            N,
            D,
            block_size,
            coords.data(),
            box.data(),
            py_bb_ctrs.mutable_data(),
            py_bb_exts.mutable_data()
        );

        return py::make_tuple(py_bb_ctrs, py_bb_exts);

    })
    .def("get_nblist", [](
        timemachine::Neighborlist<RealType> &nblist,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &box,
        const double cutoff) -> std::vector<std::vector<int> > {

        int N = coords.shape()[0];
        int D = coords.shape()[1];

        std::vector<std::vector<int> > ixn_list = nblist.get_nblist_host(
            N,
            coords.data(),
            box.data(),
            cutoff
        );

        return ixn_list;

    });


}

void declare_context(py::module &m) {

    using Class = timemachine::Context;
    std::string pyclass_name = std::string("Context");
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<double, py::array::c_style> &x0,
        const py::array_t<double, py::array::c_style> &v0,
        const py::array_t<double, py::array::c_style> &box0,
        timemachine::Integrator *intg,
        std::vector<timemachine::BoundPotential *> bps) {
        // std::vector<timemachine::Observable *> obs) {

        int N = x0.shape()[0];
        int D = x0.shape()[1];

        if(N != v0.shape()[0]) {
            throw std::runtime_error("v0 N != x0 N");
        }

        if(D != v0.shape()[1]) {
            throw std::runtime_error("v0 D != x0 D");
        }

        return new timemachine::Context(
            N,
            x0.data(),
            v0.data(),
            box0.data(),
            intg,
            bps
            // obs
        );

    }))
    .def("add_observable", &timemachine::Context::add_observable)
    .def("step", &timemachine::Context::step)
    .def("multiple_steps", [](timemachine::Context &ctxt,
        const py::array_t<double, py::array::c_style> &lambda_schedule,
        int store_du_dl_interval,
        int store_x_interval) -> py::tuple {
        // (ytz): I hate C++
        std::vector<double> vec_lambda_schedule(lambda_schedule.size());
        std::memcpy(vec_lambda_schedule.data(), lambda_schedule.data(), vec_lambda_schedule.size()*sizeof(double));

        int du_dl_interval = (store_du_dl_interval <= 0) ? lambda_schedule.size() : store_du_dl_interval;
        int x_interval = (store_x_interval <= 0) ? lambda_schedule.size() : store_x_interval;

        std::array<std::vector<double>, 2> result = ctxt.multiple_steps(vec_lambda_schedule, du_dl_interval, x_interval);

        py::array_t<double, py::array::c_style> out_du_dl_buffer(result[0].size());
        std::memcpy(out_du_dl_buffer.mutable_data(), result[0].data(), result[0].size()*sizeof(double));

        int N = ctxt.num_atoms();
        int D = 3;
        int F = (lambda_schedule.size() + x_interval - 1) / x_interval;
        py::array_t<double, py::array::c_style> out_x_buffer({F, N, D});
        std::memcpy(out_x_buffer.mutable_data(), result[1].data(), result[1].size()*sizeof(double));

        return py::make_tuple(out_du_dl_buffer, out_x_buffer);
    }, py::arg("lambda_schedule"), py::arg("store_du_dl_interval") = 0, py::arg("store_x_interval") = 0)
    // .def("multiple_steps", &timemachine::Context::multiple_steps)
    .def("get_x_t", [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
        unsigned int N = ctxt.num_atoms();
        unsigned int D = 3;
        py::array_t<double, py::array::c_style> buffer({N, D});
        ctxt.get_x_t(buffer.mutable_data());
        return buffer;
    })
    .def("get_v_t", [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
        unsigned int N = ctxt.num_atoms();
        unsigned int D = 3;
        py::array_t<double, py::array::c_style> buffer({N, D});
        ctxt.get_v_t(buffer.mutable_data());
        return buffer;
    })
    .def("_get_du_dx_t_minus_1", [](timemachine::Context &ctxt) -> py::array_t<double, py::array::c_style> {
        PyErr_WarnEx(PyExc_DeprecationWarning,
            "_get_du_dx_t_minus_1() should only be used for testing. It will be removed in a future release.",
            1);
        unsigned int N = ctxt.num_atoms();
        unsigned int D = 3;
        std::vector<unsigned long long> du_dx(N*D);
        ctxt.get_du_dx_t_minus_1(&du_dx[0]);
        py::array_t<double, py::array::c_style> py_du_dx({N, D});
        for(int i=0; i < du_dx.size(); i++) {
            py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
        }
        return py_du_dx;
    });

}

void declare_observable(py::module &m) {

    using Class = timemachine::Observable;
    std::string pyclass_name = std::string("Observable");
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    );
}

void declare_avg_partial_u_partial_param(py::module &m) {

    using Class = timemachine::AvgPartialUPartialParam;
    std::string pyclass_name = std::string("AvgPartialUPartialParam");
    py::class_<Class, timemachine::Observable>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        timemachine::BoundPotential *bp,
        int interval) {
        return new timemachine::AvgPartialUPartialParam(bp, interval);
    }))
    .def("avg_du_dp", [](timemachine::AvgPartialUPartialParam &obj) -> py::array_t<double, py::array::c_style> {
        std::vector<int> shape = obj.shape();
        py::array_t<double, py::array::c_style> buffer(shape);

        obj.avg_du_dp(buffer.mutable_data());

        return buffer;
    })
    .def("std_du_dp", [](timemachine::AvgPartialUPartialParam &obj) -> py::array_t<double, py::array::c_style> {
        std::vector<int> shape = obj.shape();
        py::array_t<double, py::array::c_style> buffer(shape);

        obj.std_du_dp(buffer.mutable_data());

        return buffer;
    });
}
void declare_integrator(py::module &m) {

    using Class = timemachine::Integrator;
    std::string pyclass_name = std::string("Integrator");
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    );
}


void declare_langevin_integrator(py::module &m) {

    using Class = timemachine::LangevinIntegrator;
    std::string pyclass_name = std::string("LangevinIntegrator");
    py::class_<Class, timemachine::Integrator>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        double dt,
        double ca,
        const py::array_t<double, py::array::c_style> &cbs,
        const py::array_t<double, py::array::c_style> &ccs,
        int seed) {

        return new timemachine::LangevinIntegrator(
            cbs.size(),
            dt,
            ca,
            cbs.data(),
            ccs.data(),
            seed
        );

    }
    ));

}

void declare_potential(py::module &m) {

    using Class = timemachine::Potential;
    std::string pyclass_name = std::string("Potential");
    py::class_<Class, std::shared_ptr<Class> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("execute", [](timemachine::Potential &pot,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params,
        const py::array_t<double, py::array::c_style> &box,
        double lambda) -> py::tuple  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.size();

            std::vector<unsigned long long> du_dx(N*D);
            std::vector<double> du_dp(P);

            // initialize to zero for the accumulator
            std::vector<unsigned long long> du_dl(N, 0);
            std::vector<unsigned long long> u(N, 0);

            pot.execute_host(
                N,
                P,
                coords.data(),
                params.data(),
                box.data(),
                lambda,
                &du_dx[0],
                &du_dp[0],
                &du_dl[0],
                &u[0]
            );

            py::array_t<double, py::array::c_style> py_du_dx({N, D});
            for(int i=0; i < du_dx.size(); i++) {
                // py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
            }

            std::vector<ssize_t> pshape(params.shape(), params.shape()+params.ndim());

            py::array_t<double, py::array::c_style> py_du_dp(pshape);
            for(int i=0; i < du_dp.size(); i++) {
                py_du_dp.mutable_data()[i] = du_dp[i];
            }

            unsigned long long du_dl_sum = std::accumulate(du_dl.begin(), du_dl.end(), decltype(du_dl)::value_type(0));
            unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

            return py::make_tuple(py_du_dx, py_du_dp, FIXED_TO_FLOAT<double>(du_dl_sum), FIXED_TO_FLOAT<double>(u_sum));

    })
    .def("execute_selective", [](timemachine::Potential &pot,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params,
        const py::array_t<double, py::array::c_style> &box,
        double lambda,
        bool compute_du_dx,
        bool compute_du_dp,
        bool compute_du_dl,
        bool compute_u) -> py::tuple  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.size();

            std::vector<unsigned long long> du_dx(N*D);
            std::vector<double> du_dp(P);

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
                compute_u ? &u[0] : nullptr
            );

            py::array_t<double, py::array::c_style> py_du_dx({N, D});
            for(int i=0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
            }

            std::vector<ssize_t> pshape(params.shape(), params.shape()+params.ndim());

            py::array_t<double, py::array::c_style> py_du_dp(pshape);
            for(int i=0; i < du_dp.size(); i++) {
                py_du_dp.mutable_data()[i] = du_dp[i];
            }

            unsigned long long du_dl_sum = std::accumulate(du_dl.begin(), du_dl.end(), decltype(du_dl)::value_type(0));
            unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

            auto result = py::make_tuple(
                py_du_dx,
                py_du_dp,
                FIXED_TO_FLOAT<double>(du_dl_sum),
                FIXED_TO_FLOAT<double>(u_sum)
            );

            if(!compute_du_dx) {
                result[0] = py::none();
            }
            if(!compute_du_dp) {
                result[1] = py::none();
            }
            if(!compute_du_dl) {
                result[2] = py::none();
            }
            if(!compute_u) {
                result[3] = py::none();
            }

            return result;
    })
    .def("execute_du_dx", [](timemachine::Potential &pot,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params,
        const py::array_t<double, py::array::c_style> &box,
        double lambda) -> py::array_t<double, py::array::c_style> {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.size();

            std::vector<unsigned long long> du_dx(N*D);

            pot.execute_host_du_dx(
                N,
                P,
                coords.data(),
                params.data(),
                box.data(),
                lambda,
                &du_dx[0]
            );

            py::array_t<double, py::array::c_style> py_du_dx({N, D});
            for(int i=0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = static_cast<double>(static_cast<long long>(du_dx[i]))/FIXED_EXPONENT;
            }

            return py_du_dx;
    });

}


void declare_bound_potential(py::module &m) {

    using Class = timemachine::BoundPotential;
    std::string pyclass_name = std::string("BoundPotential");
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def(py::init([](
        std::shared_ptr<timemachine::Potential> potential,
        const py::array_t<double, py::array::c_style> &params
    ) {

        std::vector<int> pshape(params.shape(), params.shape()+params.ndim());

        return new timemachine::BoundPotential(
            potential,
            pshape,
            params.data()
        );
    }
    ))
    .def("size", &timemachine::BoundPotential::size)
    .def("execute", [](timemachine::BoundPotential &bp,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &box,
        double lambda) -> py::tuple  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];

            std::vector<unsigned long long> du_dx(N*D);
            std::vector<unsigned long long> du_dl(N, 0);
            std::vector<unsigned long long> u(N, 0);

            bp.execute_host(
                N,
                coords.data(),
                box.data(),
                lambda,
                &du_dx[0],
                &du_dl[0],
                &u[0]
            );

            py::array_t<double, py::array::c_style> py_du_dx({N, D});
            for(int i=0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<double>(du_dx[i]);
            }

            unsigned long long du_dl_sum = std::accumulate(du_dl.begin(), du_dl.end(), decltype(du_dl)::value_type(0));
            unsigned long long u_sum = std::accumulate(u.begin(), u.end(), decltype(u)::value_type(0));

            return py::make_tuple(py_du_dx, FIXED_TO_FLOAT<double>(du_dl_sum), FIXED_TO_FLOAT<double>(u_sum));
    });

}

// template <typename RealType>
// void declare_shape(py::module &m, const char *typestr) {

//     using Class = timemachine::Shape<RealType>;
//     std::string pyclass_name = std::string("Shape_") + typestr;
//     py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
//         m,
//         pyclass_name.c_str(),
//         py::buffer_protocol(),
//         py::dynamic_attr()
//     )
//     .def(py::init([](
//         const int N,
//         const py::array_t<int, py::array::c_style> &a_idxs,
//         const py::array_t<int, py::array::c_style> &b_idxs,
//         const py::array_t<double, py::array::c_style> &alphas,
//         const py::array_t<double, py::array::c_style> &weights,
//         double k) {

//             std::vector<int> vec_a_idxs(a_idxs.data(), a_idxs.data()+a_idxs.size());
//             std::vector<int> vec_b_idxs(b_idxs.data(), b_idxs.data()+b_idxs.size());
//             std::vector<double> vec_alphas(alphas.data(), alphas.data()+alphas.size());
//             std::vector<double> vec_weights(weights.data(), weights.data()+weights.size());

//             return new timemachine::Shape<RealType>(
//                 N,
//                 vec_a_idxs,
//                 vec_b_idxs,
//                 vec_alphas,
//                 vec_weights,
//                 k
//             );

//         }
//     ));

// }


template <typename RealType>
void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<RealType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &bond_idxs,
        std::optional<py::array_t<int, py::array::c_style> > lamb_mult,
        std::optional<py::array_t<int, py::array::c_style> > lamb_offset
    ){
        std::vector<int> vec_bond_idxs(bond_idxs.data(), bond_idxs.data()+bond_idxs.size());
        std::vector<int> vec_lamb_mult;
        std::vector<int> vec_lamb_offset;
        if(lamb_mult.has_value()) {
            vec_lamb_mult.assign(lamb_mult.value().data(), lamb_mult.value().data()+lamb_mult.value().size());
        }
        if(lamb_offset.has_value()) {
            vec_lamb_offset.assign(lamb_offset.value().data(), lamb_offset.value().data()+lamb_offset.value().size());
        }
        return new timemachine::HarmonicBond<RealType>(
            vec_bond_idxs,
            vec_lamb_mult,
            vec_lamb_offset
        );
    }),
    py::arg("bond_idxs"), py::arg("lamb_mult") = py::none(), py::arg("lamb_offset") = py::none()
    );

}

template <typename RealType>
void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &angle_idxs,
        std::optional<py::array_t<int, py::array::c_style> > lamb_mult,
        std::optional<py::array_t<int, py::array::c_style> > lamb_offset
    ){
        std::vector<int> vec_angle_idxs(angle_idxs.size());
        std::memcpy(vec_angle_idxs.data(), angle_idxs.data(), vec_angle_idxs.size()*sizeof(int));
        std::vector<int> vec_lamb_mult;
        std::vector<int> vec_lamb_offset;
        if(lamb_mult.has_value()) {
            vec_lamb_mult.assign(lamb_mult.value().data(), lamb_mult.value().data()+lamb_mult.value().size());
        }
        if(lamb_offset.has_value()) {
            vec_lamb_offset.assign(lamb_offset.value().data(), lamb_offset.value().data()+lamb_offset.value().size());
        }
        return new timemachine::HarmonicAngle<RealType>(
            vec_angle_idxs,
            vec_lamb_mult,
            vec_lamb_offset
        );
    }),
    py::arg("angle_idxs"), py::arg("lamb_mult") = py::none(), py::arg("lamb_offset") = py::none()
    );

}



// template <typename RealType>
// void declare_restraint(py::module &m, const char *typestr) {

//     using Class = timemachine::Restraint<RealType>;
//     std::string pyclass_name = std::string("Restraint_") + typestr;
//     py::class_<Class, timemachine::Gradient>(
//         m,
//         pyclass_name.c_str(),
//         py::buffer_protocol(),
//         py::dynamic_attr()
//     )
//     .def(py::init([](
//         const py::array_t<int, py::array::c_style> &bond_idxs,
//         const py::array_t<double, py::array::c_style> &params,
//         const py::array_t<int, py::array::c_style> &lambda_flags
//     ){
//         std::vector<int> vec_bond_idxs(bond_idxs.size());
//         std::memcpy(vec_bond_idxs.data(), bond_idxs.data(), vec_bond_idxs.size()*sizeof(int));
//         std::vector<double> vec_params(params.size());
//         std::memcpy(vec_params.data(), params.data(), vec_params.size()*sizeof(double)); // important to use doubles
//         std::vector<int> vec_lambda_flags(lambda_flags.size());
//         std::memcpy(vec_lambda_flags.data(), lambda_flags.data(), vec_lambda_flags.size()*sizeof(int));

//         return new timemachine::Restraint<RealType>(
//             vec_bond_idxs,
//             vec_params,
//             vec_lambda_flags
//         );
//     }
//     ))
//     .def("get_du_dp_primals", [](timemachine::Restraint<RealType> &grad) -> py::array_t<double, py::array::c_style> {
//         const int B = grad.num_bonds();
//         py::array_t<double, py::array::c_style> buffer({B, 3});
//         grad.get_du_dp_primals(buffer.mutable_data());
//         return buffer;
//     })
//     .def("get_du_dp_tangents", [](timemachine::Restraint<RealType> &grad) -> py::array_t<double, py::array::c_style> {
//         const int B = grad.num_bonds();
//         py::array_t<double, py::array::c_style> buffer({B, 3});
//         grad.get_du_dp_tangents(buffer.mutable_data());
//         return buffer;
//     });

// }


template <typename RealType>
void declare_rmsd_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::RMSDRestraint<RealType>;
    std::string pyclass_name = std::string("RMSDRestraint_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &atom_map,
        const int N,
        const double k
    ) {
        std::vector<int> vec_atom_map(atom_map.size());
        std::memcpy(vec_atom_map.data(), atom_map.data(), vec_atom_map.size()*sizeof(int));

        return new timemachine::RMSDRestraint<RealType>(
            vec_atom_map,
            N,
            k
        );

    }));

}


template <typename RealType>
void declare_centroid_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::CentroidRestraint<RealType>;
    std::string pyclass_name = std::string("CentroidRestraint_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &group_a_idxs,
        const py::array_t<int, py::array::c_style> &group_b_idxs,
        double kb,
        double b0
    ) {
        std::vector<int> vec_group_a_idxs(group_a_idxs.size());
        std::memcpy(vec_group_a_idxs.data(), group_a_idxs.data(), vec_group_a_idxs.size()*sizeof(int));
        std::vector<int> vec_group_b_idxs(group_b_idxs.size());
        std::memcpy(vec_group_b_idxs.data(), group_b_idxs.data(), vec_group_b_idxs.size()*sizeof(int));

        return new timemachine::CentroidRestraint<RealType>(
            vec_group_a_idxs,
            vec_group_b_idxs,
            kb,
            b0
        );

    }));

}


// template <typename RealType>
// void declare_inertial_restraint(py::module &m, const char *typestr) {

//     using Class = timemachine::InertialRestraint<RealType>;
//     std::string pyclass_name = std::string("InertialRestraint_") + typestr;
//     py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
//         m,
//         pyclass_name.c_str(),
//         py::buffer_protocol(),
//         py::dynamic_attr()
//     )
//     .def(py::init([](
//         const py::array_t<int, py::array::c_style> &group_a_idxs,
//         const py::array_t<int, py::array::c_style> &group_b_idxs,
//         const py::array_t<double, py::array::c_style> &masses,
//         double k
//     ) {
//         std::vector<int> vec_group_a_idxs(group_a_idxs.size());
//         std::memcpy(vec_group_a_idxs.data(), group_a_idxs.data(), vec_group_a_idxs.size()*sizeof(int));
//         std::vector<int> vec_group_b_idxs(group_b_idxs.size());
//         std::memcpy(vec_group_b_idxs.data(), group_b_idxs.data(), vec_group_b_idxs.size()*sizeof(int));
//         std::vector<double> vec_masses(masses.size());
//         std::memcpy(vec_masses.data(), masses.data(), vec_masses.size()*sizeof(double));

//         return new timemachine::InertialRestraint<RealType>(
//             vec_group_a_idxs,
//             vec_group_b_idxs,
//             vec_masses,
//             k
//         );

//     }));

// }


template <typename RealType>
void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<RealType>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &torsion_idxs,
        std::optional<py::array_t<int, py::array::c_style> > lamb_mult,
        std::optional<py::array_t<int, py::array::c_style> > lamb_offset) {
        std::vector<int> vec_torsion_idxs(torsion_idxs.size());
        std::memcpy(vec_torsion_idxs.data(), torsion_idxs.data(), vec_torsion_idxs.size()*sizeof(int));
        std::vector<int> vec_lamb_mult;
        std::vector<int> vec_lamb_offset;
        if(lamb_mult.has_value()) {
            vec_lamb_mult.assign(lamb_mult.value().data(), lamb_mult.value().data()+lamb_mult.value().size());
        }
        if(lamb_offset.has_value()) {
            vec_lamb_offset.assign(lamb_offset.value().data(), lamb_offset.value().data()+lamb_offset.value().size());
        }
        return new timemachine::PeriodicTorsion<RealType>(
            vec_torsion_idxs,
            vec_lamb_mult,
            vec_lamb_offset
        );
    }),
    py::arg("angle_idxs"), py::arg("lamb_mult") = py::none(), py::arg("lamb_offset") = py::none()
    );

}

// void declare_lambda_potential(py::module &m) {

//     using Class = timemachine::LambdaPotential;
//     std::string pyclass_name = std::string("LambdaPotential");
//     py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
//         m,
//         pyclass_name.c_str(),
//         py::buffer_protocol(),
//         py::dynamic_attr()
//     )
//     .def(py::init([](
//         std::shared_ptr<timemachine::Potential> potential,
//         int N,
//         int P,
//         double multiplier,
//         double offset) {

//         return new timemachine::LambdaPotential(
//             potential,
//             N,
//             P,
//             multiplier,
//             offset
//         );

//     }
//     ));

// }


// void declare_interpolated_potential(py::module &m) {

//     using Class = timemachine::InterpolatedPotential;
//     std::string pyclass_name = std::string("InterpolatedPotential");
//     py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
//         m,
//         pyclass_name.c_str(),
//         py::buffer_protocol(),
//         py::dynamic_attr()
//     )
//     .def(py::init([](
//         std::shared_ptr<timemachine::Potential> potential,
//         int N,
//         int P) {

//         return new timemachine::InterpolatedPotential(
//             potential,
//             N,
//             P
//         );

//     }
//     ));

// }


// stackoverflow
std::string dirname(const std::string& fname) {
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}


template <typename RealType, bool Interpolated>
void declare_nonbonded(py::module &m, const char *typestr) {

    using Class = timemachine::Nonbonded<RealType, Interpolated>;
    std::string pyclass_name = std::string("Nonbonded_") + typestr;
    py::class_<Class, std::shared_ptr<Class>, timemachine::Potential>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def("set_nblist_padding", &timemachine::Nonbonded<RealType, Interpolated>::set_nblist_padding)
    .def("disable_hilbert_sort", &timemachine::Nonbonded<RealType, Interpolated>::disable_hilbert_sort)
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &exclusion_i,  // [E, 2] comprised of elements from N
        const py::array_t<double, py::array::c_style> &scales_i,  // [E, 2]
        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i, //
        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i, //
        const double beta,
        const double cutoff,
        const std::string &transform_lambda_charge="lambda",
        const std::string &transform_lambda_sigma="lambda",
        const std::string &transform_lambda_epsilon="lambda",
        const std::string &transform_lambda_w="lambda") {

        std::vector<int> exclusion_idxs(exclusion_i.size());
        std::memcpy(exclusion_idxs.data(), exclusion_i.data(), exclusion_i.size()*sizeof(int));

        std::vector<double> scales(scales_i.size());
        std::memcpy(scales.data(), scales_i.data(), scales_i.size()*sizeof(double));

        std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
        std::memcpy(lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size()*sizeof(int));

        std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
        std::memcpy(lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size()*sizeof(int));

        std::string dir_path = dirname(__FILE__);
        std::string src_path = dir_path + "/kernels/k_lambda_transformer_jit.cuh";
        std::ifstream t(src_path);
        std::string source_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_CHARGE"), transform_lambda_charge);
        source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_SIGMA"), transform_lambda_sigma);
        source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_EPSILON"), transform_lambda_epsilon);
        source_str = std::regex_replace(source_str, std::regex("CUSTOM_EXPRESSION_W"), transform_lambda_w);

        return new timemachine::Nonbonded<RealType, Interpolated>(
            exclusion_idxs,
            scales,
            lambda_plane_idxs,
            lambda_offset_idxs,
            beta,
            cutoff,
            source_str
        );
    }),
    py::arg("exclusion_i"),
    py::arg("scales_i"),
    py::arg("lambda_plane_idxs_i"),
    py::arg("lambda_offset_idxs_i"),
    py::arg("beta"),
    py::arg("cutoff"),
    py::arg("transform_lambda_charge")="return lambda",
    py::arg("transform_lambda_sigma")="return lambda",
    py::arg("transform_lambda_epsilon")="return lambda",
    py::arg("transform_lambda_w")="return lambda");

}

PYBIND11_MODULE(custom_ops, m) {

    declare_integrator(m);
    declare_langevin_integrator(m);

    declare_observable(m);
    declare_avg_partial_u_partial_param(m);
    // declare_avg_partial_u_partial_lambda(m);
    // declare_full_partial_u_partial_lambda(m);

    declare_potential(m);
    declare_bound_potential(m);
    // declare_lambda_potential(m);
    // declare_interpolated_potential(m);

    declare_neighborlist<double>(m, "f64");
    declare_neighborlist<float>(m, "f32");

    declare_centroid_restraint<double>(m, "f64");
    declare_centroid_restraint<float>(m, "f32");

    // declare_inertial_restraint<double>(m, "f64");
    // declare_inertial_restraint<float>(m, "f32");

    declare_rmsd_restraint<double>(m, "f64");
    declare_rmsd_restraint<float>(m, "f32");

    // declare_shape<double>(m, "f64");
    // declare_shape<float>(m, "f32");

    declare_harmonic_bond<double>(m, "f64");
    declare_harmonic_bond<float>(m, "f32");

    declare_harmonic_angle<double>(m, "f64");
    declare_harmonic_angle<float>(m, "f32");

    declare_periodic_torsion<double>(m, "f64");
    declare_periodic_torsion<float>(m, "f32");

    declare_nonbonded<double, true>(m, "f64_interpolated");
    declare_nonbonded<float, true>(m, "f32_interpolated");

    declare_nonbonded<double, false>(m, "f64");
    declare_nonbonded<float, false>(m, "f32");

    // declare_gbsa<double>(m, "f64");
    // declare_gbsa<float>(m, "f32");

    declare_context(m);

}
