#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "context.hpp"
#include "harmonic_bond.hpp"
#include "harmonic_angle.hpp"
#include "restraint.hpp"
#include "periodic_torsion.hpp"
#include "nonbonded.hpp"
#include "gbsa.hpp"
#include "gradient.hpp"
#include "fixed_point.hpp"

#include <iostream>

namespace py = pybind11;


void declare_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::Stepper;
    std::string pyclass_name = std::string("Stepper_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    );

}

void declare_alchemical_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::AlchemicalStepper;
    std::string pyclass_name = std::string("AlchemicalStepper_") + typestr;
    py::class_<Class, timemachine::Stepper >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const std::vector<timemachine::Gradient*> system,
        const std::vector<double> &lambda_schedule
    ) {
        return new timemachine::AlchemicalStepper(
            system,
            lambda_schedule
        );
    }))
    .def("get_du_dl", [](timemachine::AlchemicalStepper &stepper) -> py::array_t<double, py::array::c_style> {
        const unsigned long long T = stepper.get_T();
        const unsigned long long F = stepper.get_F();
        py::array_t<double, py::array::c_style> buffer({F, T});
        stepper.get_du_dl(buffer.mutable_data());
        return buffer;
    })
    .def("get_energies", [](timemachine::AlchemicalStepper &stepper) -> py::array_t<double, py::array::c_style> {
        const unsigned long long T = stepper.get_T();
        py::array_t<double, py::array::c_style> buffer({T});
        stepper.get_energies(buffer.mutable_data());
        return buffer;
    })
    .def("set_du_dl_adjoint", [](timemachine::AlchemicalStepper &stepper,
        const py::array_t<double, py::array::c_style> &adjoints) {
        stepper.set_du_dl_adjoint(adjoints.size(), adjoints.data());
    });
}

void declare_reversible_context(py::module &m, const char *typestr) {

    using Class = timemachine::ReversibleContext;
    std::string pyclass_name = std::string("ReversibleContext_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        timemachine::Stepper *stepper,
        const py::array_t<double, py::array::c_style> &x0,
        const py::array_t<double, py::array::c_style> &v0,
        const py::array_t<double, py::array::c_style> &coeff_cas,
        const py::array_t<double, py::array::c_style> &coeff_cbs,
        const py::array_t<double, py::array::c_style> &coeff_ccs,
        const py::array_t<double, py::array::c_style> &step_sizes,
        unsigned long long seed
    ) {

        int N = x0.shape()[0];
        int D = x0.shape()[1];

        if(N != v0.shape()[0]) {
            throw std::runtime_error("v0 N != x0 N");
        }

        if(D != v0.shape()[1]) {
            throw std::runtime_error("v0 D != x0 D");
        }

        int T = coeff_cas.shape()[0];

        if(T != step_sizes.shape()[0]) {
            throw std::runtime_error("coeff_cas T != step_sizes T");
        }

        // int P = params.shape()[0];

        std::vector<double> x0_vec(x0.data(), x0.data()+x0.size());
        std::vector<double> v0_vec(v0.data(), v0.data()+v0.size());
        std::vector<double> coeff_cas_vec(coeff_cas.data(), coeff_cas.data()+coeff_cas.size());
        std::vector<double> coeff_cbs_vec(coeff_cbs.data(), coeff_cbs.data()+coeff_cbs.size());
        std::vector<double> coeff_ccs_vec(coeff_ccs.data(), coeff_ccs.data()+coeff_ccs.size());
        std::vector<double> step_sizes_vec(step_sizes.data(), step_sizes.data()+step_sizes.size());
        // std::vector<double> params_vec(params.data(), params.data()+params.size());

        return new timemachine::ReversibleContext(
            stepper,
            N,
            x0_vec,
            v0_vec,
            coeff_cas_vec,
            coeff_cbs_vec,
            coeff_ccs_vec,
            step_sizes_vec,
            // params_vec,
            seed
        );

    }))
    .def("forward_mode", &timemachine::ReversibleContext::forward_mode)
    .def("backward_mode", &timemachine::ReversibleContext::backward_mode)
    // .def("get_param_adjoint_accum", [](timemachine::ReversibleContext &ctxt) -> py::array_t<double, py::array::c_style> {
    //     unsigned int P = ctxt.P();
    //     py::array_t<double, py::array::c_style> buffer({P});
    //     ctxt.get_param_adjoint_accum(buffer.mutable_data());
    //     return buffer;
    // })
    .def("get_last_coords", [](timemachine::ReversibleContext &ctxt) -> py::array_t<double, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int D = 3;
        py::array_t<double, py::array::c_style> buffer({N, D});
        ctxt.get_last_coords(buffer.mutable_data());
        return buffer;
    })
    .def("get_all_coords", [](timemachine::ReversibleContext &ctxt) -> py::array_t<double, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int F = ctxt.F();
        unsigned int D = 3;
        py::array_t<double, py::array::c_style> buffer({F, N, D});
        ctxt.get_all_coords(buffer.mutable_data());
        return buffer;
    })
    .def("set_x_t_adjoint", [](timemachine::ReversibleContext &ctxt,
        const py::array_t<double, py::array::c_style> &xt) {
        ctxt.set_x_t_adjoint(xt.data());
    })
    .def("get_x_t_adjoint", [](timemachine::ReversibleContext &ctxt) -> 
        py::array_t<double, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int D = 3;
        py::array_t<double, py::array::c_style> buffer({N, D});
        ctxt.get_x_t_adjoint(buffer.mutable_data());
        return buffer;
    })
    .def("get_v_t_adjoint", [](timemachine::ReversibleContext &ctxt) -> 
        py::array_t<double, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int D = 3;
        py::array_t<double, py::array::c_style> buffer({N, D});
        ctxt.get_v_t_adjoint(buffer.mutable_data());
        return buffer;
    });
}


void declare_gradient(py::module &m) {

    using Class = timemachine::Gradient;
    std::string pyclass_name = std::string("Gradient");
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("execute_lambda", [](timemachine::Gradient &grad,
        const py::array_t<double, py::array::c_style> &coords,
        // const py::array_t<double, py::array::c_style> &params,
        double lambda) -> py::tuple  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            // const long unsigned int P = params.shape()[0];

            std::vector<unsigned long long> out_coords(N*D);

            double out_du_dl = -9999999999; //debug use, make sure its overwritten
            double out_energy = 9999999999; //debug use, make sure its overwrriten

            grad.execute_lambda_inference_host(
                N,
                // P,
                coords.data(),
                // params.data(),
                lambda,
                &out_coords[0],
                &out_du_dl,
                &out_energy
            );

            py::array_t<double, py::array::c_style> py_out_coords({N, D});
            for(int i=0; i < out_coords.size(); i++) {
                py_out_coords.mutable_data()[i] = static_cast<double>(static_cast<long long>(out_coords[i]))/FIXED_EXPONENT;
            }

            return py::make_tuple(py_out_coords, out_du_dl, out_energy);
    })
    .def("execute_lambda_jvp", [](timemachine::Gradient &grad,
        const py::array_t<double, py::array::c_style> &coords,
        // const py::array_t<double, py::array::c_style> &params,
        double lambda,
        const py::array_t<double, py::array::c_style> &coords_tangents,
        // const py::array_t<double, py::array::c_style> &params_tangents,
        double lambda_tangent) -> py::tuple {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            // const long unsigned int P = params.shape()[0];

            py::array_t<double, py::array::c_style> py_out_coords_primals({N, D});
            py::array_t<double, py::array::c_style> py_out_coords_tangents({N, D});

            // py::array_t<double, py::array::c_style> py_out_params_primals({P});
            // py::array_t<double, py::array::c_style> py_out_params_tangents({P});

            grad.execute_lambda_jvp_host(
                N,
                coords.data(),
                coords_tangents.data(),
                // params.data(),
                lambda,
                lambda_tangent,
                py_out_coords_primals.mutable_data(),
                py_out_coords_tangents.mutable_data()
                // py_out_params_primals.mutable_data(),
                // py_out_params_tangents.mutable_data()
            );

            return py::make_tuple(py_out_coords_tangents, py_out_coords_primals);
    });

}

template <typename RealType>
void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<RealType>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &bond_idxs,
        const py::array_t<double, py::array::c_style> &params
    ){
        std::vector<int> vec_bond_idxs(bond_idxs.size());
        std::memcpy(vec_bond_idxs.data(), bond_idxs.data(), vec_bond_idxs.size()*sizeof(int));
        std::vector<double> vec_params(params.size());
        std::memcpy(vec_params.data(), params.data(), vec_params.size()*sizeof(double));

        return new timemachine::HarmonicBond<RealType>(
            vec_bond_idxs,
            vec_params
        );
    }
    ))
    .def("get_du_dp_primals", [](timemachine::HarmonicBond<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int B = grad.num_bonds();
        py::array_t<double, py::array::c_style> buffer({B, 2});
        grad.get_du_dp_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dp_tangents", [](timemachine::HarmonicBond<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int B = grad.num_bonds();
        py::array_t<double, py::array::c_style> buffer({B, 2});
        grad.get_du_dp_tangents(buffer.mutable_data());
        return buffer;
    });

}


template <typename RealType>
void declare_restraint(py::module &m, const char *typestr) {

    using Class = timemachine::Restraint<RealType>;
    std::string pyclass_name = std::string("Restraint_") + typestr;
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &bond_idxs,
        const py::array_t<double, py::array::c_style> &params,
        const py::array_t<int, py::array::c_style> &lambda_flags
    ){
        std::vector<int> vec_bond_idxs(bond_idxs.size());
        std::memcpy(vec_bond_idxs.data(), bond_idxs.data(), vec_bond_idxs.size()*sizeof(int));
        std::vector<double> vec_params(params.size());
        std::memcpy(vec_params.data(), params.data(), vec_params.size()*sizeof(double)); // important to use doubles
        std::vector<int> vec_lambda_flags(lambda_flags.size());
        std::memcpy(vec_lambda_flags.data(), lambda_flags.data(), vec_lambda_flags.size()*sizeof(int));

        return new timemachine::Restraint<RealType>(
            vec_bond_idxs,
            vec_params,
            vec_lambda_flags
        );
    }
    ))
    .def("get_du_dp_primals", [](timemachine::Restraint<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int B = grad.num_bonds();
        py::array_t<double, py::array::c_style> buffer({B, 3});
        grad.get_du_dp_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dp_tangents", [](timemachine::Restraint<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int B = grad.num_bonds();
        py::array_t<double, py::array::c_style> buffer({B, 3});
        grad.get_du_dp_tangents(buffer.mutable_data());
        return buffer;
    });;

}


template <typename RealType>
void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &angle_idxs,
        const py::array_t<double, py::array::c_style> &params_i
    ){
        std::vector<int> vec_angle_idxs(angle_idxs.size());
        std::memcpy(vec_angle_idxs.data(), angle_idxs.data(), vec_angle_idxs.size()*sizeof(int));
        std::vector<double> params(params_i.size());
        std::memcpy(params.data(), params_i.data(), params.size()*sizeof(double));

        return new timemachine::HarmonicAngle<RealType>(
            vec_angle_idxs,
            params
        );
    }
    ))
    .def("get_du_dp_primals", [](timemachine::HarmonicAngle<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int A = grad.num_angles();
        py::array_t<double, py::array::c_style> buffer({A, 2});
        grad.get_du_dp_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dp_tangents", [](timemachine::HarmonicAngle<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int A = grad.num_angles();
        py::array_t<double, py::array::c_style> buffer({A, 2});
        grad.get_du_dp_tangents(buffer.mutable_data());
        return buffer;
    });

}


template <typename RealType>
void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<RealType>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &torsion_idxs,
        const py::array_t<double, py::array::c_style> &params
    ){
        std::vector<int> vec_torsion_idxs(torsion_idxs.size());
        std::memcpy(vec_torsion_idxs.data(), torsion_idxs.data(), vec_torsion_idxs.size()*sizeof(int));
        std::vector<double> vec_params(params.size());
        std::memcpy(vec_params.data(), params.data(), vec_params.size()*sizeof(double));

        return new timemachine::PeriodicTorsion<RealType>(
            vec_torsion_idxs,
            vec_params
        );
    }
    ))
    .def("get_du_dp_primals", [](timemachine::PeriodicTorsion<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int T = grad.num_torsions();
        py::array_t<double, py::array::c_style> buffer({T, 3});
        grad.get_du_dp_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dp_tangents", [](timemachine::PeriodicTorsion<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int T = grad.num_torsions();
        py::array_t<double, py::array::c_style> buffer({T, 3});
        grad.get_du_dp_tangents(buffer.mutable_data());
        return buffer;
    });;

}


template <typename RealType>
void declare_nonbonded(py::module &m, const char *typestr) {

    using Class = timemachine::Nonbonded<RealType>;
    std::string pyclass_name = std::string("Nonbonded_") + typestr;
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<double, py::array::c_style> &charge_i,  // charge_param_idxs
        const py::array_t<double, py::array::c_style> &lj_i,  // lj_param_idxs
        const py::array_t<int, py::array::c_style> &exclusion_i,  // [E, 2] comprised of elements from N
        const py::array_t<double, py::array::c_style> &charge_scale_i,  // 
        const py::array_t<double, py::array::c_style> &lj_scale_i,  // 
        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i,  //
        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i,  //
        double cutoff) {

        std::vector<double> charge_params(charge_i.size());
        std::memcpy(charge_params.data(), charge_i.data(), charge_i.size()*sizeof(double));

        std::vector<double> lj_params(lj_i.size());
        std::memcpy(lj_params.data(), lj_i.data(), lj_i.size()*sizeof(double));

        std::vector<int> exclusion_idxs(exclusion_i.size());
        std::memcpy(exclusion_idxs.data(), exclusion_i.data(), exclusion_i.size()*sizeof(int));

        std::vector<double> charge_scales(charge_scale_i.size());
        std::memcpy(charge_scales.data(), charge_scale_i.data(), charge_scale_i.size()*sizeof(double));

        std::vector<double> lj_scales(lj_scale_i.size());
        std::memcpy(lj_scales.data(), lj_scale_i.data(), lj_scale_i.size()*sizeof(double));

        std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
        std::memcpy(lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size()*sizeof(int));

        std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
        std::memcpy(lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size()*sizeof(int));

        return new timemachine::Nonbonded<RealType>(
            charge_params,
            lj_params,
            exclusion_idxs,
            charge_scales,
            lj_scales,
            lambda_plane_idxs,
            lambda_offset_idxs,
            cutoff
        );
    }
    ))
    .def("get_du_dcharge_primals", [](timemachine::Nonbonded<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer(N);
        grad.get_du_dcharge_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dcharge_tangents", [](timemachine::Nonbonded<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer(N);
        grad.get_du_dcharge_tangents(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dlj_primals", [](timemachine::Nonbonded<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer({N, 2});
        grad.get_du_dlj_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dlj_tangents", [](timemachine::Nonbonded<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer({N, 2});
        grad.get_du_dlj_tangents(buffer.mutable_data());
        return buffer;
    });

}


template <typename RealType>
void declare_gbsa(py::module &m, const char *typestr) {

    using Class = timemachine::GBSA<RealType>;
    std::string pyclass_name = std::string("GBSA_") + typestr;
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<double, py::array::c_style> &charge_params_i, // [N]
        const py::array_t<double, py::array::c_style> &gb_params_i, // [N, 2]
        // const py::array_t<int, py::array::c_style> &radii_pi, // [N]
        // const py::array_t<int, py::array::c_style> &scale_pi, // [N]
        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i,  //
        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i,  //
        double alpha,
        double beta,
        double gamma,
        double dielectric_offset,
        double surface_tension,
        double solute_dielectric,
        double solvent_dielectric,
        double probe_radius,
        double cutoff_radii,
        double cutoff_force) {

        std::vector<double> charge_params(charge_params_i.size());
        std::memcpy(charge_params.data(), charge_params_i.data(), charge_params_i.size()*sizeof(double));
        std::vector<double> gb_params(gb_params_i.size());
        std::memcpy(gb_params.data(), gb_params_i.data(), gb_params_i.size()*sizeof(double));
        // std::vector<int> scale_factor_idxs(scale_pi.size());
        // std::memcpy(scale_factor_idxs.data(), scale_pi.data(), scale_pi.size()*sizeof(int));
        std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
        std::memcpy(lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size()*sizeof(int));
        std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
        std::memcpy(lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size()*sizeof(int));


        return new timemachine::GBSA<RealType>(
            charge_params,
            gb_params,
            // charge_param_idxs, // [N]
            // atomic_radii_idxs, // [N]
            // scale_factor_idxs, // 
            lambda_plane_idxs,
            lambda_offset_idxs,
            alpha,
            beta,
            gamma,
            dielectric_offset,
            surface_tension,
            solute_dielectric,
            solvent_dielectric,
            probe_radius,
            cutoff_radii,
            cutoff_force
        );
    }))
    .def("get_du_dcharge_primals", [](timemachine::GBSA<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer(N);
        grad.get_du_dcharge_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dcharge_tangents", [](timemachine::GBSA<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer(N);
        grad.get_du_dcharge_tangents(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dgb_primals", [](timemachine::GBSA<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer({N, 2});
        grad.get_du_dgb_primals(buffer.mutable_data());
        return buffer;
    })
    .def("get_du_dgb_tangents", [](timemachine::GBSA<RealType> &grad) -> py::array_t<double, py::array::c_style> {
        const int N = grad.num_atoms();
        py::array_t<double, py::array::c_style> buffer({N, 2});
        grad.get_du_dgb_tangents(buffer.mutable_data());
        return buffer;
    });

}

PYBIND11_MODULE(custom_ops, m) {

    declare_gradient(m);
    // declare_alchemical_gradient(m);

    declare_restraint<double>(m, "f64");
    declare_restraint<float>(m, "f32");

    declare_harmonic_bond<double>(m, "f64");
    declare_harmonic_bond<float>(m, "f32");

    declare_harmonic_angle<double>(m, "f64");
    declare_harmonic_angle<float>(m, "f32");

    declare_periodic_torsion<double>(m, "f64");
    declare_periodic_torsion<float>(m, "f32");

    declare_nonbonded<double>(m, "f64");
    declare_nonbonded<float>(m, "f32");

    declare_gbsa<double>(m, "f64");
    declare_gbsa<float>(m, "f32");

    declare_stepper(m, "f64");
    declare_alchemical_stepper(m, "f64");
    declare_reversible_context(m, "f64");

}