cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(timemachine LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

set(CUDA_ARCH sm_70 CACHE STRING "Specific CUDA architecture we're targetting.")
set(BUILD_TEST OFF CACHE STRING "Whether or not we're building for tests.")

if(BUILD_TEST)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/fixed_point_test.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/fixed_point.hpp COPYONLY)
else(BUILD_TEST)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/fixed_point_prod.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/fixed_point.hpp COPYONLY)
endif(BUILD_TEST)

# -rdc=true), as well as linking in of the device cudart libraries (-lcudadevrt
string(APPEND CMAKE_CUDA_FLAGS "-Xptxas -v -arch=${CUDA_ARCH} -O3 -lineinfo -rdc=true")
message(${CMAKE_CUDA_FLAGS})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set(CMAKE_INSTALL_PREFIX ${PARENT_DIR} CACHE PATH "Default Install Path" FORCE)
endif()

set(LIBRARY_NAME custom_ops)
set(PYBIND_SRC_DIR pybind11)


if(NOT EXISTS ${PYBIND_SRC_DIR})
  execute_process(COMMAND git clone --branch tm0.1 https://github.com/proteneer/pybind11.git ${PYBIND_SRC_DIR})
endif()

set(CUB_SRC_DIR cub)

if(NOT EXISTS ${CUB_SRC_DIR})
  execute_process(COMMAND git clone --depth 1 --branch 1.10.0 https://github.com/NVIDIA/cub.git ${CUB_SRC_DIR})
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR})


# NO_EXTRAS is needed since cuda doesn't use flto
pybind11_add_module(${LIBRARY_NAME} SHARED NO_EXTRAS
  src/fixed_point.hpp
  src/wrap_kernels.cpp
  src/solver.cpp
  src/potential.cu
  src/bound_potential.cu
  src/gpu_utils.cu
  src/observable.cu
  src/hilbert.cpp
  src/nonbonded.cu
  src/neighborlist.cu
  # src/restraint.cu
  src/harmonic_bond.cu
  src/harmonic_angle.cu
  src/periodic_torsion.cu
  src/integrator.cu
  src/shape.cu
  src/lennard_jones.cu
  src/electrostatics.cu
  # src/stepper.cu
  src/context.cu
  src/lambda_potential.cu
  src/interpolated_potential.cu
  # src/gbsa.cu
  src/centroid_restraint.cu
  src/inertial_restraint.cu
  src/kernels/k_neighborlist.cu
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(src/kernels)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/${CUB_SRC_DIR})

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 14)
target_link_libraries(${LIBRARY_NAME} -lcurand -lcudart -lcudadevrt)
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")

install(TARGETS ${LIBRARY_NAME} DESTINATION "lib")
