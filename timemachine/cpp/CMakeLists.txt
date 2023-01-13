cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_PYTHON_VERSION 3.10 CACHE STRING "Which version of python we're building wrappers against")

project(timemachine LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PythonInterp 3.10 REQUIRED)
find_package(PythonLibs 3.10 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pybind11)

set(DEBUG_BUILD "$ENV{DEBUG_BUILD}")

set(COMMON_COMPILER_FLAGS "-Wall -Wextra")
if (NOT DEBUG_BUILD)
  string(APPEND COMMON_COMPILER_FLAGS " -O3")
else()
  string(APPEND COMMON_COMPILER_FLAGS " -g")
endif()


string(APPEND CMAKE_CUDA_FLAGS "${COMMON_COMPILER_FLAGS}")

string(APPEND CMAKE_CUDA_FLAGS
  " -lineinfo"  # Disabling lineinfo reduces .so file by ~40MB
  " -Werror all-warnings"
  " -Wno-sign-compare"
  " -Wno-unused-parameter"
  " -Wreorder"
  " -Wfloat-conversion"
  " -Wno-deprecated-declarations"
  " -restrict"  # Warn if device pointers are not restricted
)
message("C++ Compiler Flags" ${CMAKE_CXX_FLAGS})
message("Cuda Compiler Flags" ${CMAKE_CUDA_FLAGS})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set(CMAKE_INSTALL_PREFIX ${PARENT_DIR} CACHE PATH "Default Install Path" FORCE)
endif()

set(LIBRARY_NAME custom_ops)

# Disable warnings on vendored files
file(
  GLOB
  VENDORED_FILES
  "src/vendored/*.c*"
  "src/vendored/*.h*"
)
set_source_files_properties(
  ${VENDORED_FILES}
  PROPERTIES
  COMPILE_OPTIONS "-w"
)


# NO_EXTRAS is needed since cuda doesn't use flto
pybind11_add_module(${LIBRARY_NAME} SHARED NO_EXTRAS
  src/fixed_point.hpp
  src/wrap_kernels.cpp
  src/potential.cu
  src/mover.cu
  src/bound_potential.cu
  src/centroid_restraint.cu
  src/gpu_utils.cu
  src/vendored/hilbert.cpp
  src/mol_utils.cpp
  src/nonbonded_common.cpp
  src/nonbonded_precomputed.cu
  src/nonbonded_all_pairs.cu
  src/nonbonded_pair_list.cu
  src/nonbonded_interaction_group.cu
  src/energy_accumulation.cu
  src/local_md_utils.cu
  src/neighborlist.cu
  src/segmented_sumexp.cu
  src/harmonic_bond.cu
  src/flat_bottom_bond.cu
  src/log_flat_bottom_bond.cu
  src/harmonic_angle.cu
  src/harmonic_angle_stable.cu
  src/periodic_torsion.cu
  src/chiral_atom_restraint.cu
  src/chiral_bond_restraint.cu
  src/langevin_integrator.cu
  src/verlet_integrator.cu
  src/context.cu
  src/barostat.cu
  src/rotations.cu
  src/hilbert_sort.cu
  src/local_md_potentials.cu
  src/rmsd_align.cpp
  src/summed_potential.cu
  src/fanout_summed_potential.cu
  src/device_buffer.cu
  src/pinned_host_buffer.cu
  src/streamed_potential_runner.cu
  src/stream_manager.cu
  src/segmented_weighted_random_sampler.cu
  src/bd_exchange_move.cu
  src/tibd_exchange_move.cu
  src/nonbonded_mol_energy.cu
  src/kernels/k_hilbert.cu
  src/kernels/k_indices.cu
  src/kernels/k_logsumexp.cu
  src/kernels/k_rotations.cu
  src/kernels/k_exchange.cu
  src/kernels/k_sampling.cu
  src/all_atom_energies.cu
  src/exchange.cu
  src/translations.cu
)

include_directories(src/kernels)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/${CUB_SRC_DIR})

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 17)

if (NOT CUDA_ARCH)
  set(CUDA_ARCH "native")
endif()
message("Cuda Arch: " ${CUDA_ARCH})
set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH}")
unset(CUDA_ARCH CACHE)

target_link_libraries(${LIBRARY_NAME} PRIVATE -lcurand)
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")

add_custom_command(
  OUTPUT custom_ops.pyi
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_stubs
  WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  DEPENDS ${LIBRARY_NAME}
  COMMENT "Generating typing stubs for ${LIBRARY_NAME}")

add_custom_target(${LIBRARY_NAME}_stubs ALL DEPENDS custom_ops.pyi)

install(TARGETS ${LIBRARY_NAME} DESTINATION "lib")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/custom_ops.pyi DESTINATION "lib")
