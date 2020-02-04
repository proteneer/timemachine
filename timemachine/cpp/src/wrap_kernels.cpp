#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "new_context.hpp"
#include "optimizer.hpp"
#include "langevin.hpp"
#include "harmonic_bond.hpp"
#include "harmonic_angle.hpp"
#include "periodic_torsion.hpp"
#include "nonbonded.hpp"
#include "gradient.hpp"
#include "stepper.hpp"
#include "fixed_point.hpp"

#include <iostream>

namespace py = pybind11;


template <typename RealType>
void declare_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::Stepper<RealType>;
    std::string pyclass_name = std::string("Stepper_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    );

}

template <typename RealType>
void declare_basic_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::BasicStepper<RealType>;
    std::string pyclass_name = std::string("BasicStepper_") + typestr;
    py::class_<Class, timemachine::Stepper<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const std::vector<timemachine::Gradient<RealType, 3> *> system
    ) {
        return new timemachine::BasicStepper<RealType>(
            system
        );
    }));
}


template <typename RealType>
void declare_lambda_stepper(py::module &m, const char *typestr) {

    using Class = timemachine::LambdaStepper<RealType>;
    std::string pyclass_name = std::string("LambdaStepper_") + typestr;
    py::class_<Class, timemachine::Stepper<RealType> >(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        const std::vector<timemachine::Gradient<RealType, 4> *> system,
        const std::vector<RealType> &lambda_schedule,
        const std::vector<int> &lambda_flags,
        const int exponent
    ) {
        return new timemachine::LambdaStepper<RealType>(
            system,
            lambda_schedule,
            lambda_flags,
            exponent
        );
    }))
    .def("get_du_dl", [](timemachine::LambdaStepper<RealType> &stepper) -> py::array_t<RealType, py::array::c_style> {
        const int T = stepper.get_T();
        py::array_t<RealType, py::array::c_style> buffer({T});
        stepper.get_du_dl(buffer.mutable_data());
        return buffer;
    })
    .def("set_du_dl_adjoint", [](timemachine::LambdaStepper<RealType> &stepper,
        const py::array_t<RealType, py::array::c_style> &adjoints) {
        stepper.set_du_dl_adjoint(adjoints.shape()[0], adjoints.data());
    })
    .def("forward_step", [](timemachine::LambdaStepper<RealType> &stepper,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params) -> py::array_t<RealType, py::array::c_style> {

        unsigned int N = coords.shape()[0];
        unsigned int D = coords.shape()[1];

        if(D != 3) {
            throw std::runtime_error("D must be 3 for lambda stepper!");
        }
        unsigned int P = params.shape()[0];

        std::vector<unsigned long long> forces(N*D);

        py::array_t<RealType, py::array::c_style> buffer({N, D});
        stepper.forward_step_host(
            N,
            P,
            coords.data(),
            params.data(),
            &forces[0]
        );

        py::array_t<RealType, py::array::c_style> py_out_coords({N, D});
        for(int i=0; i < forces.size(); i++) {
            buffer.mutable_data()[i] = static_cast<RealType>(static_cast<long long>(forces[i]))/FIXED_EXPONENT;
        }

        return buffer;

    })
    .def("backward_step", [](timemachine::LambdaStepper<RealType> &stepper,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params,
        const py::array_t<RealType, py::array::c_style> &coords_tangent) -> py::tuple {

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

        py::array_t<RealType, py::array::c_style> py_out_coords_jvp({N, D});
        py::array_t<RealType, py::array::c_style> py_out_params_jvp({P});

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


template <typename RealType, int D>
void declare_reversible_context(py::module &m, const char *typestr) {

    using Class = timemachine::ReversibleContext<RealType, D>;
    std::string pyclass_name = std::string("ReversibleContext_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr()
    )
    .def(py::init([](
        timemachine::Stepper<RealType> *stepper,
        int N,
        const std::vector<RealType> &x0,
        const std::vector<RealType> &v0,
        const std::vector<RealType> &coeff_cas,
        const std::vector<RealType> &coeff_cbs,
        const std::vector<RealType> &step_sizes,
        const std::vector<RealType> &params
    ) {

        return new timemachine::ReversibleContext<RealType, D>(
            stepper,
            N,
            x0,
            v0,
            coeff_cas,
            coeff_cbs,
            step_sizes,
            params
        );

    }))
    .def("forward_mode", &timemachine::ReversibleContext<RealType, D>::forward_mode)
    .def("backward_mode", &timemachine::ReversibleContext<RealType, D>::backward_mode)
    .def("get_param_adjoint_accum", [](timemachine::ReversibleContext<RealType, D> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        unsigned int P = ctxt.P();
        py::array_t<RealType, py::array::c_style> buffer({P});
        ctxt.get_param_adjoint_accum(buffer.mutable_data());
        return buffer;
    })
    .def("get_all_coords", [](timemachine::ReversibleContext<RealType, D> &ctxt) -> py::array_t<RealType, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int F = ctxt.F();
        unsigned int DD = D;
        py::array_t<RealType, py::array::c_style> buffer({F, N, DD});
        ctxt.get_all_coords(buffer.mutable_data());
        return buffer;
    })
    .def("set_x_t_adjoint", [](timemachine::ReversibleContext<RealType, D> &ctxt,
        const py::array_t<RealType, py::array::c_style> &xt) {
        ctxt.set_x_t_adjoint(xt.data());
    })
    .def("get_x_t_adjoint", [](timemachine::ReversibleContext<RealType, D> &ctxt) -> 
        py::array_t<RealType, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int DD = D;
        py::array_t<RealType, py::array::c_style> buffer({N, DD});
        ctxt.get_x_t_adjoint(buffer.mutable_data());
        return buffer;
    })
    .def("get_v_t_adjoint", [](timemachine::ReversibleContext<RealType, D> &ctxt) -> 
        py::array_t<RealType, py::array::c_style> {
        unsigned int N = ctxt.N();
        unsigned int DD = D;
        py::array_t<RealType, py::array::c_style> buffer({N, DD});
        ctxt.get_v_t_adjoint(buffer.mutable_data());
        return buffer;
    });
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
        })
    .def("get_dt", &timemachine::Optimizer<RealType>::get_dt);

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
        const py::array_t<RealType, py::array::c_style> &cc,  // param_idxs
        const int n_offset // optimization for speedups
    ) {
        std::vector<RealType> coeff_bs(cb.size());
        std::memcpy(coeff_bs.data(), cb.data(), cb.size()*sizeof(RealType));
        std::vector<RealType> coeff_cs(cc.size());
        std::memcpy(coeff_cs.data(), cc.data(), cc.size()*sizeof(RealType));
        return new timemachine::LangevinOptimizer<RealType>(dt, ndims, ca, coeff_bs, coeff_cs, n_offset);
    }),
        py::arg("dt").none(false),
        py::arg("ndims").none(false),
        py::arg("ca").none(false),
        py::arg("cb").none(false),
        py::arg("cc").none(false),
        py::arg("n_offset").none(false)
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
    })
    .def("set_coeff_d", [](timemachine::LangevinOptimizer<RealType> &lo,
        const RealType cd) {
        lo.set_coeff_d(cd);
    })
    .def("set_coeff_e", [](timemachine::LangevinOptimizer<RealType> &lo,
        const RealType ce) {
        lo.set_coeff_e(ce);
    });


}


template <typename RealType, int D>
void declare_gradient(py::module &m, const char *typestr) {

    using Class = timemachine::Gradient<RealType, D>;
    std::string pyclass_name = std::string("Gradient_") + typestr;
    py::class_<Class>(
        m,
        pyclass_name.c_str(),
        py::buffer_protocol(),
        py::dynamic_attr())
    .def("execute", [](timemachine::Gradient<RealType, D> &grad,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params) -> py::array_t<RealType, py::array::c_style>  {

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

            py::array_t<RealType, py::array::c_style> py_out_coords({N, DD});
            for(int i=0; i < out_coords.size(); i++) {
                py_out_coords.mutable_data()[i] = static_cast<RealType>(static_cast<long long>(out_coords[i]))/FIXED_EXPONENT;
            }

            return py_out_coords;
    })
   .def("execute_jvp", [](timemachine::Gradient<RealType, D> &grad,
        const py::array_t<RealType, py::array::c_style> &coords,
        const py::array_t<RealType, py::array::c_style> &params,
        const py::array_t<RealType, py::array::c_style> &coords_tangents,
        const py::array_t<RealType, py::array::c_style> &params_tangents) -> py::tuple {

            const long unsigned int N = coords.shape()[0];
            const long unsigned int DD = coords.shape()[1];

            if(DD != D) throw std::runtime_error("D mismatch");

            const long unsigned int P = params.shape()[0];

            py::array_t<RealType, py::array::c_style> py_out_coords_tangents({N, DD});
            py::array_t<RealType, py::array::c_style> py_out_params_tangents({P});

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
    py::class_<Class, timemachine::Gradient<RealType, D> >(
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

        return new timemachine::HarmonicBond<RealType, D>(
            vec_bond_idxs,
            vec_param_idxs
        );
    }
    ));

}


template <typename RealType, int D>
void declare_harmonic_angle(py::module &m, const char *typestr) {

    using Class = timemachine::HarmonicAngle<RealType, D>;
    std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
    py::class_<Class, timemachine::Gradient<RealType, D> >(
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
    py::class_<Class, timemachine::Gradient<RealType, D> >(
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
    py::class_<Class, timemachine::Gradient<RealType, D> >(
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
        double cutoff
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
            cutoff
        );
    }
    ));

}


PYBIND11_MODULE(custom_ops, m) {

    declare_gradient<double, 4>(m, "f64_4d");
    declare_gradient<double, 3>(m, "f64_3d");

    declare_harmonic_bond<double, 4>(m, "f64_4d");
    declare_harmonic_bond<double, 3>(m, "f64_3d");

    declare_harmonic_angle<double, 4>(m, "f64_4d");
    declare_harmonic_angle<double, 3>(m, "f64_3d");

    declare_periodic_torsion<double, 4>(m, "f64_4d");
    declare_periodic_torsion<double, 3>(m, "f64_3d");

    declare_nonbonded<double, 4>(m, "f64_4d");
    declare_nonbonded<double, 3>(m, "f64_3d");

    declare_stepper<double>(m, "f64");
    declare_basic_stepper<double>(m, "f64");
    declare_lambda_stepper<double>(m, "f64");

    declare_reversible_context<double, 4>(m, "f64_4d");
    declare_reversible_context<double, 3>(m, "f64_3d");

    declare_optimizer<float>(m, "f32");
    declare_optimizer<double>(m, "f64");

    declare_langevin_optimizer<float>(m, "f32");
    declare_langevin_optimizer<double>(m, "f64");

}