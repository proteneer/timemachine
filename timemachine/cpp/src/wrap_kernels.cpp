#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "context.hpp"
#include "optimizer.hpp"
#include "langevin.hpp"
#include "harmonic_bond.hpp"
#include "harmonic_angle.hpp"
#include "periodic_torsion.hpp"
#include "nonbonded.hpp"
#include "gbsa.hpp"
#include "gradient.hpp"
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

void declare_basic_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::BasicStepper;
    std::string pyclass_name = std::string("BasicStepper_") + typestr;
    py::class_<Class, timemachine::Stepper >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const std::vector<timemachine::Gradient<3> *> system
    ) {
        return new timemachine::BasicStepper(
            system
        );
    }));
}


void declare_lambda_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::LambdaStepper;
    std::string pyclass_name = std::string("LambdaStepper_") + typestr;
    py::class_<Class, timemachine::Stepper >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const std::vector<timemachine::Gradient<4> *> system,
        const std::vector<double> &lambda_schedule,
        const std::vector<int> &lambda_flags
    ) {
        return new timemachine::LambdaStepper(
            system,
            lambda_schedule,
            lambda_flags
        );
    }))
    .def("get_du_dl", [](timemachine::LambdaStepper &stepper) -> py::array_t<double, py::array::c_style> {
        const unsigned long long T = stepper.get_T();
        py::array_t<double, py::array::c_style> buffer({T});
        stepper.get_du_dl(buffer.mutable_data());
        return buffer;
    })
    .def("set_du_dl_adjoint", [](timemachine::LambdaStepper &stepper,
        const py::array_t<double, py::array::c_style> &adjoints) {
        stepper.set_du_dl_adjoint(adjoints.shape()[0], adjoints.data());
    })
    .def("forward_step", [](timemachine::LambdaStepper &stepper,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params) -> py::array_t<double, py::array::c_style> {

        unsigned int N = coords.shape()[0];
        unsigned int D = coords.shape()[1];

        if(D != 3) {
            throw std::runtime_error("D must be 3 for lambda stepper!");
        }
        unsigned int P = params.shape()[0];

        std::vector<unsigned long long> forces(N*D);

        py::array_t<double, py::array::c_style> buffer({N, D});
        stepper.forward_step_host(
            N,
            P,
            coords.data(),
            params.data(),
            &forces[0]
        );

        py::array_t<double, py::array::c_style> py_out_coords({N, D});
        for(int i=0; i < forces.size(); i++) {
            buffer.mutable_data()[i] = static_cast<double>(static_cast<long long>(forces[i]))/FIXED_EXPONENT;
        }

        return buffer;

    })
    .def("backward_step", [](timemachine::LambdaStepper &stepper,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params,
        const py::array_t<double, py::array::c_style> &coords_tangent) -> py::tuple {

        unsigned int N = coords.shape()[0];
        unsigned int D = coords.shape()[1];

        if(coords_tangent.shape()[0] != N) {
            throw std::runtime_error("tangent shape mismatch N");
        }
        if(coords_tangent.shape()[1] != D) {
            throw std::runtime_error("tangent shape mismatch N");
        }

        if(D != 3) {
            throw std::runtime_error("D must be 3 for lambda stepper!");
        }
        unsigned int P = params.shape()[0];

        py::array_t<double, py::array::c_style> py_out_coords_jvp({N, D});
        py::array_t<double, py::array::c_style> py_out_params_jvp({P});

        stepper.backward_step_host(
            N,
            P,
            coords.data(),
            params.data(),
            coords_tangent.data(),
            py_out_coords_jvp.mutable_data(),
            py_out_params_jvp.mutable_data()
        );

        return py::make_tuple(py_out_coords_jvp, py_out_params_jvp);

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


template <int D>
void declare_gradient(py::module &m, const char *typestr) {

    using Class = timemachine::Gradient<D>;
    std::string pyclass_name = std::string("Gradient_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("execute_lambda", [](timemachine::Gradient<D> &grad,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params,
        double lambda) -> py::tuple  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int DD = coords.shape()[1];

            if(DD != D) throw std::runtime_error("D mismatch");

            const long unsigned int P = params.shape()[0];

            std::vector<unsigned long long> out_coords(N*DD);

            double out_du_dl = -9999999999; //debug use, make sure its overwritten

            grad.execute_lambda_host(
                N,P,
                coords.data(),
                nullptr,
                params.data(),
                lambda,
                0,
                &out_coords[0],
                &out_du_dl,
                nullptr,
                nullptr
            );

            py::array_t<double, py::array::c_style> py_out_coords({N, DD});
            for(int i=0; i < out_coords.size(); i++) {
                py_out_coords.mutable_data()[i] = static_cast<double>(static_cast<long long>(out_coords[i]))/FIXED_EXPONENT;
            }

            return py::make_tuple(py_out_coords, out_du_dl);
    })
    .def("execute", [](timemachine::Gradient<D> &grad,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params) -> py::array_t<double, py::array::c_style>  {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int DD = coords.shape()[1];

            if(DD != D) throw std::runtime_error("D mismatch");

            const long unsigned int P = params.shape()[0];

            std::vector<unsigned long long> out_coords(N*DD);

            grad.execute_host(
                N,P,
                coords.data(),
                nullptr,
                params.data(),
                &out_coords[0],
                nullptr,
                nullptr
            );

            py::array_t<double, py::array::c_style> py_out_coords({N, DD});
            for(int i=0; i < out_coords.size(); i++) {
                py_out_coords.mutable_data()[i] = static_cast<double>(static_cast<long long>(out_coords[i]))/FIXED_EXPONENT;
            }

            return py_out_coords;
    })
   .def("execute_jvp", [](timemachine::Gradient<D> &grad,
        const py::array_t<double, py::array::c_style> &coords,
        const py::array_t<double, py::array::c_style> &params,
        const py::array_t<double, py::array::c_style> &coords_tangents,
        const py::array_t<double, py::array::c_style> &params_tangents) -> py::tuple {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int DD = coords.shape()[1];

            if(DD != D) throw std::runtime_error("D mismatch");

            const long unsigned int P = params.shape()[0];

            py::array_t<double, py::array::c_style> py_out_coords_tangents({N, DD});
            py::array_t<double, py::array::c_style> py_out_params_tangents({P});

            grad.execute_host(
                N,P,
                coords.data(),
                coords_tangents.data(),
                params.data(),
                nullptr,
                py_out_coords_tangents.mutable_data(),
                py_out_params_tangents.mutable_data()
            );

            return py::make_tuple(py_out_coords_tangents, py_out_params_tangents);
    });

}


template <typename RealType, int D>
void declare_harmonic_bond(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicBond<RealType, D>;
    std::string pyclass_name = std::string("HarmonicBond_") + typestr;
    py::class_<Class, timemachine::Gradient<D> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &bond_idxs,
        const py::array_t<int, py::array::c_style> &param_idxs,
        const py::array_t<int, py::array::c_style> &lambda_idxs
    ){
        std::vector<int> vec_bond_idxs(bond_idxs.size());
        std::memcpy(vec_bond_idxs.data(), bond_idxs.data(), vec_bond_idxs.size()*sizeof(int));
        std::vector<int> vec_param_idxs(param_idxs.size());
        std::memcpy(vec_param_idxs.data(), param_idxs.data(), vec_param_idxs.size()*sizeof(int));
        std::vector<int> vec_lambda_idxs(lambda_idxs.size());
        std::memcpy(vec_lambda_idxs.data(), lambda_idxs.data(), vec_lambda_idxs.size()*sizeof(int));

        return new timemachine::HarmonicBond<RealType, D>(
            vec_bond_idxs,
            vec_param_idxs,
            vec_lambda_idxs
        );
    }
    ));

}


template <typename RealType, int D>
void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType, D>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, timemachine::Gradient<D> >(
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

        return new timemachine::HarmonicAngle<RealType, D>(
            vec_angle_idxs,
            vec_param_idxs
        );
    }
    ));

}


template <typename RealType, int D>
void declare_periodic_torsion(py::module &m, const char *typestr) {

    using Class = timemachine::PeriodicTorsion<RealType, D>;
    std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
    py::class_<Class, timemachine::Gradient<D> >(
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

        return new timemachine::PeriodicTorsion<RealType, D>(
            vec_torsion_idxs,
            vec_param_idxs
        );
    }
    ));

}


template <typename RealType, int D>
void declare_nonbonded(py::module &m, const char *typestr) {

    using Class = timemachine::Nonbonded<RealType, D>;
    std::string pyclass_name = std::string("Nonbonded_") + typestr;
    py::class_<Class, timemachine::Gradient<D> >(
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
        double cutoff,
        int N_limit
    ){
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

        return new timemachine::Nonbonded<RealType, D>(
            charge_param_idxs,
            lj_param_idxs,
            exclusion_idxs,
            charge_scale_idxs,
            lj_scale_idxs,
            cutoff,
            N_limit
        );
    }
    ));

}


template <typename RealType, int D>
void declare_gbsa(py::module &m, const char *typestr) {

    using Class = timemachine::GBSA<RealType, D>;
    std::string pyclass_name = std::string("GBSA_") + typestr;
    py::class_<Class, timemachine::Gradient<D> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const py::array_t<int, py::array::c_style> &charge_pi, // [N]
        const py::array_t<int, py::array::c_style> &radii_pi, // [N]
        const py::array_t<int, py::array::c_style> &scale_pi, // [N]
        double alpha,
        double beta,
        double gamma,
        double dielectric_offset,
        double surface_tension,
        double solute_dielectric,
        double solvent_dielectric,
        double probe_radius,
        double cutoff_radii,
        double cutoff_force,
        int N_limit
    ){
        std::vector<int> charge_param_idxs(charge_pi.size());
        std::memcpy(charge_param_idxs.data(), charge_pi.data(), charge_pi.size()*sizeof(int));
        std::vector<int> atomic_radii_idxs(radii_pi.size());
        std::memcpy(atomic_radii_idxs.data(), radii_pi.data(), radii_pi.size()*sizeof(int));
        std::vector<int> scale_factor_idxs(scale_pi.size());
        std::memcpy(scale_factor_idxs.data(), scale_pi.data(), scale_pi.size()*sizeof(int));

        return new timemachine::GBSA<RealType, D>(
            charge_param_idxs, // [N]
            atomic_radii_idxs, // [N]
            scale_factor_idxs, // 
            alpha,
            beta,
            gamma,
            dielectric_offset,
            surface_tension,
            solute_dielectric,
            solvent_dielectric,
            probe_radius,
            cutoff_radii,
            cutoff_force,
            N_limit
        );
    }
    ));

}

PYBIND11_MODULE(custom_ops, m) {

    declare_gradient<3>(m, "f64_3d");
    declare_gradient<4>(m, "f64_4d");

    declare_harmonic_bond<double, 4>(m, "f64_4d");
    declare_harmonic_bond<double, 3>(m, "f64_3d");
    declare_harmonic_bond<float, 4>(m, "f32_4d");
    declare_harmonic_bond<float, 3>(m, "f32_3d");

    // declare_harmonic_angle<double, 4>(m, "f64_4d");
    // declare_harmonic_angle<double, 3>(m, "f64_3d");
    // declare_harmonic_angle<float, 4>(m, "f32_4d");
    // declare_harmonic_angle<float, 3>(m, "f32_3d");

    // declare_periodic_torsion<double, 4>(m, "f64_4d");
    // declare_periodic_torsion<double, 3>(m, "f64_3d");
    // declare_periodic_torsion<float, 4>(m, "f32_4d");
    // declare_periodic_torsion<float, 3>(m, "f32_3d");

    // declare_nonbonded<double, 4>(m, "f64_4d");
    // declare_nonbonded<double, 3>(m, "f64_3d");
    // declare_nonbonded<float, 4>(m, "f32_4d");
    // declare_nonbonded<float, 3>(m, "f32_3d");

    // declare_gbsa<double, 4>(m, "f64_4d");
    // declare_gbsa<double, 3>(m, "f64_3d");
    // declare_gbsa<float, 4>(m, "f32_4d");
    // declare_gbsa<float, 3>(m, "f32_3d");

    // declare_stepper(m, "f64");
    // declare_basic_stepper(m, "f64");
    // declare_lambda_stepper(m, "f64");
    // declare_reversible_context(m, "f64_3d");

}