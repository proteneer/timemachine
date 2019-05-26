#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gpu/potential.hpp"
#include "gpu/custom_bonded_gpu.hpp"
#include "gpu/custom_nonbonded_gpu.hpp"

#include<iostream>

namespace py = pybind11;


template <typename RealType>
void declare_potential(py::module &m, const char *typestr) {

    using Class = timemachine::Potential<RealType>;
    std::string pyclass_name = std::string("Potential") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("derivatives", [](timemachine::Potential<RealType> &nrg,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params,
        const py::array_t<RealType, py::array::c_style> &dx_dp,
        const py::array_t<int, py::array::c_style> &dp_idxs) -> py::tuple {

            const long unsigned int num_confs = coords.shape()[0];
            const long unsigned int num_atoms = coords.shape()[1];
            const long unsigned int num_dims = coords.shape()[2];
            const long unsigned int num_params = params.shape()[0];
            const long unsigned int num_dp_idxs = dp_idxs.shape()[0];

            py::array_t<RealType, py::array::c_style> py_E({num_confs});
            py::array_t<RealType, py::array::c_style> py_dE_dp({num_confs, num_dp_idxs});
            py::array_t<RealType, py::array::c_style> py_dE_dx({num_confs, num_atoms, num_dims});
            py::array_t<RealType, py::array::c_style> py_d2E_dxdp({num_confs, num_dp_idxs, num_atoms, num_dims});

            memset(py_E.mutable_data(), 0.0, sizeof(RealType)*num_confs);
            memset(py_dE_dp.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_dp_idxs);
            memset(py_dE_dx.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_atoms*num_dims);
            memset(py_d2E_dxdp.mutable_data(), 0.0, sizeof(RealType)*num_confs*num_dp_idxs*num_atoms*num_dims);

            RealType *dx_dp_ptr = nullptr;

            if(dx_dp.size() > 0) {
                // (ytz):this is a safe const_cast since the resulting pointer gets passed
                // immediately into a function that takes in a const RealType * again
                dx_dp_ptr = const_cast<RealType *>(dx_dp.data());
            } 

            nrg.derivatives_host(
                num_confs,
                num_atoms,
                num_params,
                coords.data(),
                params.data(),
                py_E.mutable_data(),
                py_dE_dx.mutable_data(),
                dx_dp_ptr,
                dp_idxs.data(),
                num_dp_idxs,
                py_dE_dp.mutable_data(),
                py_d2E_dxdp.mutable_data()
            );

            return py::make_tuple(py_E, py_dE_dx, py_dE_dp, py_d2E_dxdp);
        }, 
            py::arg("coords").none(false),
            py::arg("params").none(false),
            py::arg("dx_dp").none(false),
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

PYBIND11_MODULE(custom_ops, m) {

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

}