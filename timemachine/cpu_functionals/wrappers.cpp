#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "integrator.hpp"
#include "nonbonded_cpu.hpp"
#include "nonbonded_gpu.hpp"
#include "bonded_cpu.hpp"
#include "bonded_gpu.hpp"
#include "context.hpp"
#include "energy.hpp"


#include <iostream>
#include <ctime>


namespace py = pybind11;

template <typename NumericType>
void declare_energy_gpu(py::module &m, const char *typestr) {

    using Class = timemachine::EnergyGPU<NumericType>;
    std::string pyclass_name = std::string("EnergyGPU_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());

}


// needs to subclass!
template<typename NumericType>
void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<NumericType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<size_t> // bond_idxs
    >())
    .def("total_derivative", [](timemachine::HarmonicBond<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"bonded: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}


template<typename NumericType>
void declare_harmonic_bond_gpu(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBondGPU<NumericType>;
    std::string pyclass_name = std::string("HarmonicBondGPU_") + typestr;
    py::class_<Class, timemachine::EnergyGPU<NumericType> >(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<size_t> // bond_idxs
    >())
    .def("total_derivative", [](timemachine::HarmonicBondGPU<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative_cpu(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"bonded: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}

template<typename NumericType>
void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<NumericType>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<size_t>, // angle_idxs
        bool // whether or not we use cosine angles
    >())
    .def("total_derivative", [](timemachine::HarmonicAngle<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"angle: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}


template<typename NumericType>
void declare_harmonic_angle_gpu(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngleGPU<NumericType>;
    std::string pyclass_name = std::string("HarmonicAngleGPU_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<size_t> // angle_idxs
    >())
    .def("total_derivative", [](timemachine::HarmonicAngleGPU<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative_cpu(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"bonded: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}


template<typename NumericType>
void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<NumericType>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<size_t>  // torsion_idxs
    >())
    .def("total_derivative", [](timemachine::PeriodicTorsion<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"torsion: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}



template<typename NumericType>
void declare_periodic_torsion_gpu(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsionGPU<NumericType>;
    std::string pyclass_name = std::string("PeriodicTorsionGPU_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<size_t> // angle_idxs
    >())
    .def("total_derivative", [](timemachine::PeriodicTorsionGPU<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative_cpu(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"bonded: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}


template<typename NumericType>
void declare_electrostatics(py::module &m, const char *typestr) {

    using Class = timemachine::Electrostatics<NumericType>;
    std::string pyclass_name = std::string("Electrostatics_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<NumericType>  // NxN scale_matrix
    >())
    .def("total_derivative", [](timemachine::Electrostatics<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);


        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"es: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}

template<typename NumericType>
void declare_lennard_jones(py::module &m, const char *typestr) {

    using Class = timemachine::LennardJones<NumericType>;
    std::string pyclass_name = std::string("LennardJones_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<NumericType>  // NxN scale_matrix
    >())
    .def("total_derivative", [](timemachine::LennardJones<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"lj612: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}


template<typename NumericType>
void declare_electrostatics_gpu(py::module &m, const char *typestr) {

    using Class = timemachine::ElectrostaticsGPU<NumericType>;
    std::string pyclass_name = std::string("ElectrostaticsGPU_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<NumericType>  // NxN scale_matrix
    >())
    .def("total_derivative", [](timemachine::ElectrostaticsGPU<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();


        nrg.total_derivative_cpu(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"es GPU: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}

template<typename NumericType>
void declare_lennard_jones_gpu(py::module &m, const char *typestr) {

    using Class = timemachine::LennardJonesGPU<NumericType>;
    std::string pyclass_name = std::string("LennardJonesGPU_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<NumericType>, // params
        std::vector<size_t>, // global_param_idxs
        std::vector<size_t>, // param_idxs
        std::vector<NumericType>  // NxN scale_matrix
    >())
    .def("total_derivative", [](timemachine::LennardJonesGPU<NumericType> &nrg,
        const py::array_t<NumericType, py::array::c_style> coords,
        ssize_t num_params) -> py::tuple {

        const auto num_atoms = coords.shape()[0];
        const auto num_dims = coords.shape()[1];

        NumericType energy = 0;
        py::array_t<NumericType, py::array::c_style> py_grads({num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_hessians({num_atoms, num_dims, num_atoms, num_dims});
        py::array_t<NumericType, py::array::c_style> py_mps({num_params, num_atoms, num_dims});

        memset(py_grads.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims);
        memset(py_hessians.mutable_data(), 0.0, sizeof(NumericType)*num_atoms*num_dims*num_atoms*num_dims);
        memset(py_mps.mutable_data(), 0.0, sizeof(NumericType)*num_params*num_atoms*num_dims);

        std::clock_t start; double duration; start = std::clock();

        nrg.total_derivative_cpu(
            num_atoms,
            num_params,
            coords.data(),
            &energy,
            py_grads.mutable_data(),
            py_hessians.mutable_data(),
            py_mps.mutable_data()
        );

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"lj GPU: "<< duration <<'\n';

        return py::make_tuple(energy, py_grads, py_hessians, py_mps);
    });

}



template<typename NumericType>
void declare_integrator(py::module &m, const char *typestr) {

    using Class = timemachine::Integrator<NumericType>;
    std::string pyclass_name = std::string("Integrator_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        NumericType, // dt
        int, // W,
        int, // N,
        int, // P,
        NumericType, // coeff_a, 
        std::vector<NumericType>,  // coeff_Bs
        std::vector<NumericType>  // coeff_Css
    >())
    .def("step", [](timemachine::Integrator<NumericType> &intg,
    const py::array_t<NumericType, py::array::c_style> grads,
    const py::array_t<NumericType, py::array::c_style> hessians,
    const py::array_t<NumericType, py::array::c_style> mixed_partials) -> void {
        intg.step_cpu(grads.data(), hessians.data(), mixed_partials.data());
    })
    .def("get_dxdp", [](timemachine::Integrator<NumericType> &intg) -> std::vector<NumericType> {
        return intg.get_dxdp();
    })
    .def("get_noise", [](timemachine::Integrator<NumericType> &intg) -> std::vector<NumericType> {
        return intg.get_noise();
    })
    .def("get_coordinates", [](timemachine::Integrator<NumericType> &intg) -> std::vector<NumericType> {
        return intg.get_coordinates();
    })
    .def("get_velocities", [](timemachine::Integrator<NumericType> &intg) -> std::vector<NumericType> {
        return intg.get_velocities();
    })
    .def("set_coordinates", [](timemachine::Integrator<NumericType> &intg, std::vector<NumericType> arg) -> void {
        return intg.set_coordinates(arg);
    })
    .def("set_velocities", [](timemachine::Integrator<NumericType> &intg, std::vector<NumericType> arg) -> void {
        return intg.set_velocities(arg);
    });

}

template<typename NumericType>
void declare_context(py::module &m, const char *typestr) {

    using Class = timemachine::Context<NumericType>;
    std::string pyclass_name = std::string("Context_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<
        std::vector<timemachine::EnergyGPU<NumericType> *>, // dt
        timemachine::Integrator<NumericType>*
    >())
    .def("step", &timemachine::Context<NumericType>::step);
}



PYBIND11_MODULE(custom_ops, m) {


declare_energy_gpu<double>(m, "double");
declare_energy_gpu<float>(m, "float");

declare_harmonic_bond<double>(m, "double");
declare_harmonic_bond<float>(m, "float");

declare_harmonic_bond_gpu<double>(m, "double");
declare_harmonic_bond_gpu<float>(m, "float");

declare_harmonic_angle<double>(m, "double");
declare_harmonic_angle<float>(m, "float");

declare_harmonic_angle_gpu<double>(m, "double");
declare_harmonic_angle_gpu<float>(m, "float");

declare_periodic_torsion<double>(m, "double");
declare_periodic_torsion<float>(m, "float");

declare_periodic_torsion_gpu<double>(m, "double");
declare_periodic_torsion_gpu<float>(m, "float");

declare_electrostatics<double>(m, "double");
declare_electrostatics<float>(m, "float");

declare_electrostatics_gpu<double>(m, "double");
declare_electrostatics_gpu<float>(m, "float");

declare_lennard_jones<double>(m, "double");
declare_lennard_jones<float>(m, "float");

declare_lennard_jones_gpu<double>(m, "double");
declare_lennard_jones_gpu<float>(m, "float");

declare_integrator<double>(m, "double");
declare_integrator<float>(m, "float");

declare_context<double>(m, "double");
declare_context<float>(m, "float");

}