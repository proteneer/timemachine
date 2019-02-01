#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "bonded_cpu.hpp"
#include <iostream>

namespace py = pybind11;


template<typename NumericType>
void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<NumericType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // param_idxs
        std::vector<size_t> // bond_idxs
    >())
    .def("total_derivative", [](timemachine::HarmonicBond<NumericType> &hb,
        const py::array_t<NumericType, py::array::c_style> coords,
        const py::array_t<NumericType, py::array::c_style> dxdp) -> py::tuple {

        const auto num_params = dxdp.shape()[0];
        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_totals({num_params, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        memset(py_totals.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);
        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);

        hb.total_derivative(
            num_atoms,
            num_params,
            coords.data(),
            dxdp.data(),
            &energy,
            py_grads.mutable_data(),
            py_totals.mutable_data()
        );
        // std::cout << "done" << std::endl;
        // std::memcpy(py_hessians.mutable_data(), &hessians[0], hessian_size);
        return py::make_tuple(energy, py_grads, py_totals);
    });

}

PYBIND11_MODULE(energy, m) {

declare_harmonic_bond<double>(m, "double");
declare_harmonic_bond<float>(m, "float");

}