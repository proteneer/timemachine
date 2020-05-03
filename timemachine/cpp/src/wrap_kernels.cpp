#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "context.hpp"
// #include "optimizer.hpp"
// #include "langevin.hpp"
#include "harmonic_bond.hpp"
#include "harmonic_angle.hpp"
#include "periodic_torsion.hpp"
#include "nonbonded.hpp"
#include "gbsa.hpp"
#include "gradient.hpp"
#include "alchemical_gradient.hpp"
#include "stepper.hpp"
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
        py::array_t<double, py::array::c_style> buffer({T});
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
        stepper.set_du_dl_adjoint(adjoints.shape()[0], adjoints.data());
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
        // int N,
        const py::array_t<double, py::array::c_style> &x0,
        const py::array_t<double, py::array::c_style> &v0,
        const py::array_t<double, py::array::c_style> &coeff_cas,
        const py::array_t<double, py::array::c_style> &coeff_cbs,
        const py::array_t<double, py::array::c_style> &coeff_ccs,
        const py::array_t<double, py::array::c_style> &step_sizes,
        const py::array_t<double, py::array::c_style> &params,
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

        int P = params.shape()[0];

        std::vector<double> x0_vec(x0.data(), x0.data()+x0.size());
        std::vector<double> v0_vec(v0.data(), v0.data()+v0.size());
        std::vector<double> coeff_cas_vec(coeff_cas.data(), coeff_cas.data()+coeff_cas.size());
        std::vector<double> coeff_cbs_vec(coeff_cbs.data(), coeff_cbs.data()+coeff_cbs.size());
        std::vector<double> coeff_ccs_vec(coeff_ccs.data(), coeff_ccs.data()+coeff_ccs.size());
        std::vector<double> step_sizes_vec(step_sizes.data(), step_sizes.data()+step_sizes.size());
        std::vector<double> params_vec(params.data(), params.data()+params.size());

        return new timemachine::ReversibleContext(
            stepper,
            N,
            x0_vec,
            v0_vec,
            coeff_cas_vec,
            coeff_cbs_vec,
            coeff_ccs_vec,
            step_sizes_vec,
            params_vec,
            seed
        );

    }))
    .def("forward_mode", &timemachine::ReversibleContext::forward_mode)
    .def("backward_mode", &timemachine::ReversibleContext::backward_mode)
    .def("get_param_adjoint_accum", [](timemachine::ReversibleContext &ctxt) -> py::array_t<double, py::array::c_style> {
        unsigned int P = ctxt.P();
        py::array_t<double, py::array::c_style> buffer({P});
        ctxt.get_param_adjoint_accum(buffer.mutable_data());
        return buffer;
    })
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
        const py::array_t<double, py::array::c_style> &params,
        double lambda) -> py::tuple  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.shape()[0];

            std::vector<unsigned long long> out_coords(N*D);

            double out_du_dl = -9999999999; //debug use, make sure its overwritten
            double out_energy = 9999999999; //debug use, make sure its overwrriten

            grad.execute_lambda_inference_host(
                N,
                P,
                coords.data(),
                params.data(),
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
        const py::array_t<double, py::array::c_style> &params,
        double lambda,
        const py::array_t<double, py::array::c_style> &coords_tangents,
        const py::array_t<double, py::array::c_style> &params_tangents,
        double lambda_tangent) -> py::tuple {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.shape()[0];

            py::array_t<double, py::array::c_style> py_out_coords_primals({N, D});
            py::array_t<double, py::array::c_style> py_out_coords_tangents({N, D});

            py::array_t<double, py::array::c_style> py_out_params_primals({P});
            py::array_t<double, py::array::c_style> py_out_params_tangents({P});

            grad.execute_lambda_jvp_host(
                N,P,
                coords.data(),
                coords_tangents.data(),
                params.data(),
                lambda,
                lambda_tangent,
                py_out_coords_primals.mutable_data(),
                py_out_coords_tangents.mutable_data(),
                py_out_params_primals.mutable_data(),
                py_out_params_tangents.mutable_data()
            );

            return py::make_tuple(py_out_coords_tangents, py_out_params_tangents, py_out_coords_primals, py_out_params_primals);
    });

}

void declare_alchemical_gradient(py::module &m) {

    using Class = timemachine::AlchemicalGradient;
    std::string pyclass_name = std::string("AlchemicalGradient");
    py::class_<Class, timemachine::Gradient>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        int N,
        int P,
        timemachine::Gradient *u0,
        timemachine::Gradient *u1
    ){
        return new timemachine::AlchemicalGradient(
            N,
            P,
            u0,
            u1
        );
    }
    ));

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
        const py::array_t<int, py::array::c_style> &param_idxs
    ){
        std::vector<int> vec_bond_idxs(bond_idxs.size());
        std::memcpy(vec_bond_idxs.data(), bond_idxs.data(), vec_bond_idxs.size()*sizeof(int));
        std::vector<int> vec_param_idxs(param_idxs.size());
        std::memcpy(vec_param_idxs.data(), param_idxs.data(), vec_param_idxs.size()*sizeof(int));

        return new timemachine::HarmonicBond<RealType>(
            vec_bond_idxs,
            vec_param_idxs
        );
    }
    ));

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
        const py::array_t<int, py::array::c_style> &param_idxs
    ){
        std::vector<int> vec_angle_idxs(angle_idxs.size());
        std::memcpy(vec_angle_idxs.data(), angle_idxs.data(), vec_angle_idxs.size()*sizeof(int));
        std::vector<int> vec_param_idxs(param_idxs.size());
        std::memcpy(vec_param_idxs.data(), param_idxs.data(), vec_param_idxs.size()*sizeof(int));

        return new timemachine::HarmonicAngle<RealType>(
            vec_angle_idxs,
            vec_param_idxs
        );
    }
    ));

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
        const py::array_t<int, py::array::c_style> &param_idxs
    ){
        std::vector<int> vec_torsion_idxs(torsion_idxs.size());
        std::memcpy(vec_torsion_idxs.data(), torsion_idxs.data(), vec_torsion_idxs.size()*sizeof(int));
        std::vector<int> vec_param_idxs(param_idxs.size());
        std::memcpy(vec_param_idxs.data(), param_idxs.data(), vec_param_idxs.size()*sizeof(int));

        return new timemachine::PeriodicTorsion<RealType>(
            vec_torsion_idxs,
            vec_param_idxs
        );
    }
    ));

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
        const py::array_t<int, py::array::c_style> &charge_pi,  // charge_param_idxs
        const py::array_t<int, py::array::c_style> &lj_pi,  // lj_param_idxs
        const py::array_t<int, py::array::c_style> &exclusion_i,  // [E, 2] comprised of elements from N
        const py::array_t<int, py::array::c_style> &charge_scale_i,  // 
        const py::array_t<int, py::array::c_style> &lj_scale_i,  // 
        const py::array_t<int, py::array::c_style> &lambda_plane_idxs_i,  //
        const py::array_t<int, py::array::c_style> &lambda_offset_idxs_i,  //
        double cutoff){
        std::vector<int> charge_param_idxs(charge_pi.size());
        std::memcpy(charge_param_idxs.data(), charge_pi.data(), charge_pi.size()*sizeof(int));
        std::vector<int> lj_param_idxs(lj_pi.size());
        std::memcpy(lj_param_idxs.data(), lj_pi.data(), lj_pi.size()*sizeof(int));

        std::vector<int> exclusion_idxs(exclusion_i.size());
        std::memcpy(exclusion_idxs.data(), exclusion_i.data(), exclusion_i.size()*sizeof(int));

        std::vector<int> charge_scale_idxs(charge_scale_i.size());
        std::memcpy(charge_scale_idxs.data(), charge_scale_i.data(), charge_scale_i.size()*sizeof(int));

        std::vector<int> lj_scale_idxs(lj_scale_i.size());
        std::memcpy(lj_scale_idxs.data(), lj_scale_i.data(), lj_scale_i.size()*sizeof(int));

        std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
        std::memcpy(lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size()*sizeof(int));

        std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
        std::memcpy(lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size()*sizeof(int));

        return new timemachine::Nonbonded<RealType>(
            charge_param_idxs,
            lj_param_idxs,
            exclusion_idxs,
            charge_scale_idxs,
            lj_scale_idxs,
            lambda_plane_idxs,
            lambda_offset_idxs,
            cutoff
        );
    }
    ));

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
        const py::array_t<int, py::array::c_style> &charge_pi, // [N]
        const py::array_t<int, py::array::c_style> &radii_pi, // [N]
        const py::array_t<int, py::array::c_style> &scale_pi, // [N]
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
        double cutoff_force
    ){
        std::vector<int> charge_param_idxs(charge_pi.size());
        std::memcpy(charge_param_idxs.data(), charge_pi.data(), charge_pi.size()*sizeof(int));
        std::vector<int> atomic_radii_idxs(radii_pi.size());
        std::memcpy(atomic_radii_idxs.data(), radii_pi.data(), radii_pi.size()*sizeof(int));
        std::vector<int> scale_factor_idxs(scale_pi.size());
        std::memcpy(scale_factor_idxs.data(), scale_pi.data(), scale_pi.size()*sizeof(int));
        std::vector<int> lambda_plane_idxs(lambda_plane_idxs_i.size());
        std::memcpy(lambda_plane_idxs.data(), lambda_plane_idxs_i.data(), lambda_plane_idxs_i.size()*sizeof(int));
        std::vector<int> lambda_offset_idxs(lambda_offset_idxs_i.size());
        std::memcpy(lambda_offset_idxs.data(), lambda_offset_idxs_i.data(), lambda_offset_idxs_i.size()*sizeof(int));


        return new timemachine::GBSA<RealType>(
            charge_param_idxs, // [N]
            atomic_radii_idxs, // [N]
            scale_factor_idxs, // 
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
    }
    ));

}

PYBIND11_MODULE(custom_ops, m) {

    declare_gradient(m);
    declare_alchemical_gradient(m);

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