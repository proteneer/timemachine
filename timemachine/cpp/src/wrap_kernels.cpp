#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "context.hpp"
#include "optimizer.hpp"
#include "langevin.hpp"
#include "potential.hpp"
#include "custom_bonded_gpu.hpp"
#include "custom_nonbonded_gpu.hpp"

#include <iostream>

namespace py = pybind11;

template <typename RealType>
void declare_context(py::module &m, const char *typestr) {

    using Class = timemachine::Context<RealType>;
    std::string pyclass_name = std::string("Context_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const std::vector<timemachine::Potential<RealType> *> system,
        const timemachine::Optimizer<RealType> *optimizer,
        const py::array_t<RealType, py::array::c_style> &params,
        const py::array_t<RealType, py::array::c_style> &x0,
        const py::array_t<RealType, py::array::c_style> &v0,
        const py::array_t<int, py::array::c_style> &dp_idxs
    ) {
        const int N = x0.shape()[0];
        const int D = x0.shape()[1];

        if(D != v0.shape()[1]) {
            throw std::runtime_error("x0 dim != v0 dim");
        }

        const int P = params.shape()[0];
        const int DP = dp_idxs.size();

        std::vector<int> gather_param_idxs(P, -1);
        for(int i=0; i < DP; i++) {
            if(gather_param_idxs[dp_idxs.data()[i]] != -1) {
                throw std::runtime_error("dp_idxs must contain only unique indices.");
            }
            gather_param_idxs[dp_idxs.data()[i]] = i;
        }


        return new timemachine::Context<RealType>(
            system,
            optimizer,
            params.data(),
            x0.data(),
            v0.data(),
            N,
            D,
            P,
            gather_param_idxs.data(),
            DP
        );

    }))
    .def("step", &timemachine::Context<RealType>::step)
    .def("get_E", [](timemachine::Context<RealType> &ctxt) -> RealType {
        RealType E;
        ctxt.get_E(&E);
        return E;
    })
    // .def("debug_compute_dE_dx", [](timemachine::Context<RealType> &ctxt,
    //     const py::array_t<RealType, py::array::c_style> &host_x) -> py::tuple {
    //     auto N = ctxt.num_atoms();
    //     py::array_t<RealType, py::array::c_style> out_E({1});
    //     py::array_t<RealType, py::array::c_style> out_dE_dx({N, 3});
    //     ctxt.debug_compute_dE_dx(host_x.data(), out_E.mutable_data(), out_dE_dx.mutable_data());
    //     return py::make_tuple(out_E, out_dE_dx);
    // })
    .def("get_dE_dx", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        auto N = ctxt.num_atoms();
        auto D = ctxt.num_dims();
        py::array_t<RealType, py::array::c_style> buffer({N, D});
        ctxt.get_dE_dx(buffer.mutable_data());
        return buffer;
    })
    .def("get_dE_dp", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        unsigned int DP = ctxt.num_dparams();
        py::array_t<RealType, py::array::c_style> buffer({DP});
        ctxt.get_dE_dp(buffer.mutable_data());
        return buffer;
    })
    .def("get_x", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        auto N = ctxt.num_atoms();
        auto D = ctxt.num_dims();
        py::array_t<RealType, py::array::c_style> buffer({N, D});
        ctxt.get_x(buffer.mutable_data());
        return buffer;
    })
    .def("get_v", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        auto N = ctxt.num_atoms();
        auto D = ctxt.num_dims();
        py::array_t<RealType, py::array::c_style> buffer({N, D});
        ctxt.get_v(buffer.mutable_data());
        return buffer;
    })
    .def("get_dx_dp", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        auto DP = ctxt.num_dparams();
        auto N = ctxt.num_atoms();
        auto D = ctxt.num_dims();
        py::array_t<RealType, py::array::c_style> buffer({DP, N, D});
        ctxt.get_dx_dp(buffer.mutable_data());
        return buffer;
    })
    .def("get_dv_dp", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        auto DP = ctxt.num_dparams();
        auto N = ctxt.num_atoms();
        auto D = ctxt.num_dims();
        py::array_t<RealType, py::array::c_style> buffer({DP, N, D});
        ctxt.get_dv_dp(buffer.mutable_data());
        return buffer;
    })
    .def("get_d2E_dx2", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        auto N = ctxt.num_atoms();
        auto D = ctxt.num_dims();
        py::array_t<RealType, py::array::c_style> buffer({N, D, N, D});
        ctxt.get_d2E_dx2(buffer.mutable_data());
        return buffer;
    });
    // .def("get_d2E_dxdp", [](timemachine::Context<RealType> &ctxt) -> py::array_t<RealType, py::array::c_style> {
    //     auto DP = ctxt.num_dparams();
    //     auto N = ctxt.num_atoms();
    //     auto D = ctxt.num_dims();
    //     py::array_t<RealType, py::array::c_style> buffer({DP, N, D});
    //     ctxt.get_d2E_dxdp(buffer.mutable_data());
    //     return buffer;
    // });

}


template <typename RealType>
void declare_optimizer(py::module &m, const char *typestr) {

    using Class = timemachine::Optimizer<RealType>;
    std::string pyclass_name = std::string("Optimizer_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("step", [](timemachine::Optimizer<RealType> &opt,
        const py::array_t<RealType, py::array::c_style> &dE_dx,
        const py::array_t<RealType, py::array::c_style> &d2E_dx2,
        const py::array_t<RealType, py::array::c_style> &d2E_dxdp,
        py::array_t<RealType, py::array::c_style> &x_t,
        py::array_t<RealType, py::array::c_style> &v_t,
        py::array_t<RealType, py::array::c_style> &dx_dp_t,
        py::array_t<RealType, py::array::c_style> &dv_dp_t,
        const py::array_t<RealType, py::array::c_style> &noise_buffer) {

            const long unsigned int num_atoms = dE_dx.shape()[0];
            const long unsigned int num_dims = dE_dx.shape()[1];
            const long unsigned int num_params = d2E_dxdp.shape()[0];

            opt.step_host(
                num_atoms,
                num_dims,
                num_params,
                dE_dx.data(),
                d2E_dx2.data(),
                d2E_dxdp.data(),
                x_t.mutable_data(),
                v_t.mutable_data(),
                dx_dp_t.mutable_data(),
                dv_dp_t.mutable_data(),
                noise_buffer.data()
            );
        });

}


template<typename RealType>
void declare_langevin_optimizer(py::module &m, const char *typestr) {

    using Class = timemachine::LangevinOptimizer<RealType>;
    std::string pyclass_name = std::string("LangevinOptimizer_") + typestr;
    py::class_<Class, timemachine::Optimizer<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const RealType dt,
        const int ndims,
        const RealType ca,
        const py::array_t<RealType, py::array::c_style> &cb, // bond_idxs
        const py::array_t<RealType, py::array::c_style> &cc  // param_idxs
    ) {
        std::vector<RealType> coeff_bs(cb.size());
        std::memcpy(coeff_bs.data(), cb.data(), cb.size()*sizeof(RealType));
        std::vector<RealType> coeff_cs(cc.size());
        std::memcpy(coeff_cs.data(), cc.data(), cc.size()*sizeof(RealType));
        return new timemachine::LangevinOptimizer<RealType>(dt, ndims, ca, coeff_bs, coeff_cs);
    }),
        py::arg("dt").none(false),
        py::arg("ndims").none(false),
        py::arg("ca").none(false),
        py::arg("cb").none(false),
        py::arg("cc").none(false)
    )
    .def("set_dt", [](timemachine::LangevinOptimizer<RealType> &lo,
        const RealType dt) {
        lo.set_dt(dt);
    })
    .def("set_coeff_a", [](timemachine::LangevinOptimizer<RealType> &lo,
        const RealType ca) {
        lo.set_coeff_a(ca);
    })
    .def("set_coeff_b", [](timemachine::LangevinOptimizer<RealType> &lo,
        const py::array_t<RealType, py::array::c_style> &cb) {
        lo.set_coeff_b(cb.shape()[0], cb.data());
    })
    .def("set_coeff_c", [](timemachine::LangevinOptimizer<RealType> &lo,
        const py::array_t<RealType, py::array::c_style> &cc) {
        lo.set_coeff_c(cc.shape()[0], cc.data());
    });


}

template <typename RealType>
void declare_potential(py::module &m, const char *typestr) {

    using Class = timemachine::Potential<RealType>;
    std::string pyclass_name = std::string("Potential_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("derivatives", [](timemachine::Potential<RealType> &nrg,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params,
        const py::array_t<int, py::array::c_style> &dp_idxs) -> py::tuple {

            const long unsigned int num_confs = coords.shape()[0];
            const long unsigned int num_atoms = coords.shape()[1];
            const long unsigned int num_dims = coords.shape()[2];

            const long unsigned int num_params = params.shape()[0];
            const long unsigned int num_dp_idxs = dp_idxs.shape()[0];

            py::array_t<RealType, py::array::c_style> py_E({num_confs});
            py::array_t<RealType, py::array::c_style> py_dE_dp({num_confs, num_dp_idxs});
            py::array_t<RealType, py::array::c_style> py_dE_dx({num_confs, num_atoms, num_dims});
            py::array_t<RealType, py::array::c_style> py_d2E_dx2({num_confs, num_atoms, num_dims, num_atoms, num_dims});
            py::array_t<RealType, py::array::c_style> py_d2E_dxdp({num_confs, num_dp_idxs, num_atoms, num_dims});

            memset(py_E.mutable_data(), 0.0, sizeof(RealType)*num_confs);
            memset(py_dE_dp.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_dp_idxs);
            memset(py_dE_dx.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_atoms*num_dims);
            memset(py_d2E_dx2.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_atoms*num_dims*num_atoms*num_dims);
            memset(py_d2E_dxdp.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_dp_idxs*num_atoms*num_dims);

            std::vector<int> gather_param_idxs(num_params, -1);
            for(size_t i=0; i < num_dp_idxs; i++) {
                if(gather_param_idxs[dp_idxs.data()[i]] != -1) {
                    throw std::runtime_error("dp_idxs must contain only unique indices.");
                }
                gather_param_idxs[dp_idxs.data()[i]] = i;
            }

            // for(size_t i=0; i < gather_param_idxs.size(); i++) {
            //     std::cout <<  "debug " << i << " " << gather_param_idxs[i] << std::endl;
            // }

            nrg.derivatives_host(
                num_confs,
                num_atoms,
                num_dims,
                num_params,
                coords.data(),
                params.data(),
                py_E.mutable_data(),
                py_dE_dx.mutable_data(),
                py_d2E_dx2.mutable_data(),
                num_dp_idxs,
                &gather_param_idxs[0],
                py_dE_dp.mutable_data(),
                py_d2E_dxdp.mutable_data()
            );

            return py::make_tuple(py_E, py_dE_dx, py_d2E_dx2, py_dE_dp, py_d2E_dxdp);
        }, 
            py::arg("coords").none(false),
            py::arg("params").none(false),
            py::arg("dp_idxs").none(false)
        );

}


template<typename RealType>
void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<RealType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class, timemachine::Potential<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &bi, // bond_idxs
        const py::array_t<int, py::array::c_style> &pi  // param_idxs
    ) {
        std::vector<int> bond_idxs(bi.size());
        std::memcpy(bond_idxs.data(), bi.data(), bi.size()*sizeof(int));
        std::vector<int> param_idxs(pi.size());
        std::memcpy(param_idxs.data(), pi.data(), pi.size()*sizeof(int));
        return new timemachine::HarmonicBond<RealType>(bond_idxs, param_idxs);
    }));

}


template<typename RealType>
void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, timemachine::Potential<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &ai, // bond_idxs
        const py::array_t<int, py::array::c_style> &pi  // param_idxs
    ) {
        std::vector<int> angle_idxs(ai.size());
        std::memcpy(angle_idxs.data(), ai.data(), ai.size()*sizeof(int));
        std::vector<int> param_idxs(pi.size());
        std::memcpy(param_idxs.data(), pi.data(), pi.size()*sizeof(int));
        return new timemachine::HarmonicAngle<RealType>(angle_idxs, param_idxs);
    }));

}


template<typename RealType>
void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<RealType>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class, timemachine::Potential<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &ti, // torsion_idxs
        const py::array_t<int, py::array::c_style> &pi  // param_idxs
    ) {

        std::vector<int> torsion_idxs(ti.size());
        std::memcpy(torsion_idxs.data(), ti.data(), ti.size()*sizeof(int));
        std::vector<int> param_idxs(pi.size());
        std::memcpy(param_idxs.data(), pi.data(), pi.size()*sizeof(int));

        return new timemachine::PeriodicTorsion<RealType>(torsion_idxs, param_idxs);
    }));

}


template<typename RealType>
void declare_lennard_jones(py::module &m, const char *typestr) {

    using Class = timemachine::LennardJones<RealType>;
    std::string pyclass_name = std::string("LennardJones_") + typestr;
    py::class_<Class, timemachine::Potential<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<RealType, py::array::c_style> &sm, // scale_matrix
        const py::array_t<int, py::array::c_style> &pi  // param_idxs
    ) {

        std::vector<RealType> scale_matrix(sm.size());
        std::memcpy(scale_matrix.data(), sm.data(), sm.size()*sizeof(RealType));
        std::vector<int> param_idxs(pi.size());
        std::memcpy(param_idxs.data(), pi.data(), pi.size()*sizeof(int));

        return new timemachine::LennardJones<RealType>(scale_matrix, param_idxs);
    }));

}

template<typename RealType>
void declare_electrostatics(py::module &m, const char *typestr) {

    using Class = timemachine::Electrostatics<RealType>;
    std::string pyclass_name = std::string("Electrostatics_") + typestr;
    py::class_<Class, timemachine::Potential<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<RealType, py::array::c_style> &sm, // scale_matrix
        const py::array_t<int, py::array::c_style> &pi  // param_idxs
    ) {

        std::vector<RealType> scale_matrix(sm.size());
        std::memcpy(scale_matrix.data(), sm.data(), sm.size()*sizeof(RealType));
        std::vector<int> param_idxs(pi.size());
        std::memcpy(param_idxs.data(), pi.data(), pi.size()*sizeof(int));

        return new timemachine::Electrostatics<RealType>(scale_matrix, param_idxs);
    }));

}

PYBIND11_MODULE(custom_ops, m) {

    // context

    declare_context<float>(m, "f32");
    declare_context<double>(m, "f64");

    // optimizers

    declare_optimizer<float>(m, "f32");
    declare_optimizer<double>(m, "f64");

    declare_langevin_optimizer<float>(m, "f32");
    declare_langevin_optimizer<double>(m, "f64");

    // potentials

    declare_potential<float>(m, "f32");
    declare_potential<double>(m, "f64");

    declare_harmonic_bond<float>(m, "f32");
    declare_harmonic_bond<double>(m, "f64");

    declare_harmonic_angle<float>(m, "f32");
    declare_harmonic_angle<double>(m, "f64");

    declare_periodic_torsion<float>(m, "f32");
    declare_periodic_torsion<double>(m, "f64");

    declare_lennard_jones<float>(m, "f32");
    declare_lennard_jones<double>(m, "f64");

    declare_electrostatics<float>(m, "f32");
    declare_electrostatics<double>(m, "f64");

}