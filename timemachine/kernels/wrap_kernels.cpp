#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gpu/potential.hpp"
#include "gpu/custom_bonded_gpu.hpp"

namespace py = pybind11;


template <typename RealType>
void declare_potential(py::module &m, const char *typestr) {

    using Class = timemachine::Potential<RealType>;
    std::string pyclass_name = std::string("Potential") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());

}

#include<iostream>

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
        const py::array_t<int, py::array::c_style> &pi// param_idxs
    ) {
        std::vector<int> bond_idxs(bi.size());
        std::memcpy(bond_idxs.data(), bi.data(), bi.size()*sizeof(int));
        std::vector<int> param_idxs(pi.size());
        std::memcpy(param_idxs.data(), pi.data(), pi.size()*sizeof(int));
        return new timemachine::HarmonicBond<RealType>(bond_idxs, param_idxs);
    }))
    .def("derivatives", [](timemachine::HarmonicBond<RealType> &nrg,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params,
        const py::array_t<RealType, py::array::c_style> &dxdps
        ) -> py::tuple {

        const long unsigned int num_atoms = coords.shape()[0];
        const long unsigned int num_dims = coords.shape()[1];
        const long unsigned int num_params = params.shape()[0];

        py::array_t<RealType, py::array::c_style> py_E({1});
        py::array_t<RealType, py::array::c_style> py_dE_dp({num_params});
        py::array_t<RealType, py::array::c_style> py_dE_dx({num_atoms, num_dims});
        py::array_t<RealType, py::array::c_style> py_d2E_dxdp({num_params, num_atoms, num_dims});

        memset(py_E.mutable_data(), 0.0, sizeof(RealType));
        memset(py_dE_dp.mutable_data(), 0.0, sizeof(RealType)*num_params);
        memset(py_dE_dx.mutable_data(), 0.0, sizeof(RealType)*num_atoms*num_dims);
        memset(py_d2E_dxdp.mutable_data(), 0.0, sizeof(RealType)*num_params*num_atoms*num_dims);

        RealType *dxdps_ptr = nullptr;

        if(dxdps.size() > 0) {

            // this is dangerously safe since it then gets passed into a function
            // that takes in a const RealType * again
            dxdps_ptr = const_cast<RealType *>(dxdps.data());
            std::cout << "SETTING ACTUAL DATA" << dxdps.size() << " " << dxdps.shape() << " " << dxdps.data()[0] << std::endl;
        } 

        nrg.derivatives_host(
            num_atoms,
            num_params,
            coords.data(),
            params.data(),
            dxdps_ptr,
            py_E.mutable_data(),
            py_dE_dp.mutable_data(),
            py_dE_dx.mutable_data(),
            py_d2E_dxdp.mutable_data()
        );

        return py::make_tuple(py_E, py_dE_dp, py_dE_dx, py_d2E_dxdp);
    },
    py::arg("coords").none(false),
    py::arg("params").none(false),
    py::arg("dxdps").none(true));

}

PYBIND11_MODULE(custom_ops, m) {

declare_potential<float>(m, "f32");
declare_potential<double>(m, "f64");

declare_harmonic_bond<float>(m, "f32");
declare_harmonic_bond<double>(m, "f64");

}