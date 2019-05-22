#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cpu/bonded_kernels.hpp"

namespace py = pybind11;

// hessian matrix product
template<typename RealType>
py::tuple declare_harmonic_bond_hmp(
    const py::array_t<RealType, py::array::c_style> &coords,
    const py::array_t<RealType, py::array::c_style> &params,
    const py::array_t<RealType, py::array::c_style> &dxdps,
    py::array_t<int, py::array::c_style> bond_idxs,
    py::array_t<int, py::array::c_style> param_idxs) {

    const auto num_atoms = coords.shape()[0];
    const auto num_dims = coords.shape()[1];
    const auto num_params = params.shape()[0];
    const int num_bonds = bond_idxs.shape()[0];
    const auto total_params = dxdps.shape()[0];

    if(total_params != num_params) {
        throw std::runtime_error("Shape inconsistency between dxdps and params");
    }

    const RealType step_size = 1e-7;
    std::vector<std::complex<RealType> > complex_params(coords.size());

    for(int p=0; p < num_params; p++) {
        complex_params[p] = std::complex<RealType>(params.data()[p], 0);
    }

    std::vector<std::complex<RealType> > hmp_grads(num_params*num_atoms*num_dims, std::complex<RealType>(0, 0));

    for(int p=0; p < num_params; p++) {
        complex_params[p] = std::complex<RealType>(complex_params[p].real(), step_size);
        std::vector<std::complex<RealType> > complex_coords(coords.size());
        for(int i=0; i < coords.size(); i++) {
            complex_coords[i] = std::complex<RealType>(
                coords.data()[i],
                dxdps.data()[p*num_atoms*num_dims+i]*step_size
            );
        }

        harmonic_bond_grad(
            num_atoms,
            num_params,
            complex_coords.data(),
            complex_params.data(),
            num_bonds,
            bond_idxs.data(),
            param_idxs.data(),
            &hmp_grads[0] + p*num_atoms*num_dims
        );
        // restore p
        complex_params[p] = std::complex<RealType>(complex_params[p].real(), 0.0);
    }

    py::array_t<RealType, py::array::c_style> py_grads({num_atoms, num_dims});
    memset(py_grads.mutable_data(), 0.0, sizeof(RealType)*num_atoms*num_dims);
    for(int i=0; i < py_grads.size(); i++) {
        py_grads.mutable_data()[i] = hmp_grads[i].real();
    }

    py::array_t<RealType, py::array::c_style> py_hvp({num_params, num_atoms, num_dims});
    memset(py_hvp.mutable_data(), 0.0, sizeof(RealType)*num_params*num_atoms*num_dims);
    for(size_t i=0; i < hmp_grads.size(); i++) {
        py_hvp.mutable_data()[i] = hmp_grads[i].imag()/step_size;
    }

    // also return real part
    return py::make_tuple(py_grads, py_hvp);

}


template<typename NumericType>
void harmonic_bond_hmp_gpu(
    const int num_atoms,
    const int num_params,
    const NumericType *coords,
    const NumericType *params,
    const NumericType *dxdps,
    const int num_bonds,
    const int *bond_idxs,
    const int *param_idxs,
    NumericType *grads,
    NumericType *hmps);


template<typename RealType>
py::tuple declare_harmonic_bond_hmp_gpu(
    const py::array_t<RealType, py::array::c_style> &coords,
    const py::array_t<RealType, py::array::c_style> &params,
    const py::array_t<RealType, py::array::c_style> &dxdps,
    py::array_t<int, py::array::c_style> bond_idxs,
    py::array_t<int, py::array::c_style> param_idxs) {

    const auto num_atoms = coords.shape()[0];
    const auto num_dims = coords.shape()[1];
    const auto num_params = params.shape()[0];
    const int num_bonds = bond_idxs.shape()[0];
    const auto total_params = dxdps.shape()[0];

    py::array_t<RealType, py::array::c_style> py_hvp({num_params, num_atoms, num_dims});
    memset(py_hvp.mutable_data(), 0.0, sizeof(RealType)*num_params*num_atoms*num_dims);

    py::array_t<RealType, py::array::c_style> py_grad({num_atoms, num_dims});
    memset(py_grad.mutable_data(), 0.0, sizeof(RealType)*num_atoms*num_dims);

    harmonic_bond_hmp_gpu<RealType>(
        num_atoms,
        num_params,
        coords.data(),
        params.data(),
        dxdps.data(),
        num_bonds,
        bond_idxs.data(),
        param_idxs.data(),
        py_grad.mutable_data(),
        py_hvp.mutable_data()
    );

    // also return real part
    return py::make_tuple(py_grad, py_hvp);

}

PYBIND11_MODULE(custom_ops, m) {

m.def("harmonic_bond_hmp_r32", &declare_harmonic_bond_hmp<float>, "hbg_r32");
m.def("harmonic_bond_hmp_r64", &declare_harmonic_bond_hmp<double>, "hbg_r64");

m.def("harmonic_bond_hmp_gpu_r32", &declare_harmonic_bond_hmp_gpu<float>, "hbg_gpu_r32");
m.def("harmonic_bond_hmp_gpu_r64", &declare_harmonic_bond_hmp_gpu<double>, "hbg_gpu_r64");

// m.def("harmonic_bond_grad_r32",
//     &declare_harmonic_bond_grad<float>,
//     "hbg_r32"
// );
// m.def("harmonic_bond_grad_r64",
//     &declare_harmonic_bond_grad<double>,
//     "hbg_r64"
// );
// m.def("harmonic_bond_grad_c64",
//     &declare_harmonic_bond_grad<std::complex<float> >,
//     "hbg_c64"
// );
// m.def("harmonic_bond_grad_c128",
//     &declare_harmonic_bond_grad<std::complex<double> >,
//     "hbg_c128"
// );

}


// g++ -O3 -march=native -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` wrap_kernels.cpp -o custom_ops`python3-config --extension-suffix`