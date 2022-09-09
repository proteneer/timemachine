cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "Which version of python we're building wrappers against")

project(timemachine LANGUAGES CXX CUDA)

find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

string(APPEND CMAKE_CXX_FLAGS "-Werror -Wall -Wextra")

string(APPEND CMAKE_CUDA_FLAGS "-O3 -lineinfo")
string(APPEND CMAKE_CUDA_FLAGS
  " -Werror all-warnings"
  " -Wall"
  " -Wno-sign-compare"
  " -Wextra"
  " -Wno-unused-parameter"
  " -Wreorder"
)
message(${CMAKE_CUDA_FLAGS})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set(CMAKE_INSTALL_PREFIX ${PARENT_DIR} CACHE PATH "Default Install Path" FORCE)
endif()

set(LIBRARY_NAME custom_ops)
set(PYBIND_SRC_DIR pybind11)

if(NOT EXISTS ${PYBIND_SRC_DIR})
  execute_process(COMMAND git clone --branch v2.6 https://github.com/pybind/pybind11.git ${PYBIND_SRC_DIR})
endif()

set(EIGEN_SRC_DIR eigen)
if(NOT EXISTS ${EIGEN_SRC_DIR})
  execute_process(COMMAND git clone --branch 3.3.9 https://gitlab.com/libeigen/eigen.git ${EIGEN_SRC_DIR})
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR})


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
  src/bound_potential.cu
  src/centroid_restraint.cu
  src/gpu_utils.cu
  src/vendored/hilbert.cpp
  src/nonbonded_precomputed.cu
  src/nonbonded_all_pairs.cu
  src/nonbonded_pair_list.cu
  src/nonbonded_interaction_group.cu
  src/neighborlist.cu
  src/harmonic_bond.cu
  src/flat_bottom_bond.cu
  src/harmonic_angle.cu
  src/periodic_torsion.cu
  src/chiral_atom_restraint.cu
  src/chiral_bond_restraint.cu
  src/langevin_integrator.cu
  src/verlet_integrator.cu
  src/context.cu
  src/barostat.cu
  src/rmsd_align.cpp
  src/summed_potential.cu
  src/fanout_summed_potential.cu
  src/device_buffer.cu
  src/kernels/k_nonbonded.cu
  src/kernels/nonbonded_common.cu
)

include_directories(src/kernels)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/${EIGEN_SRC_DIR})
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/${CUB_SRC_DIR})

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 17)

if (NOT CUDA_ARCH)
  find_program(HAS_SMI "nvidia-smi")
  # In cuda 11.6 nvidia-smi added support to get GPU compute capability query
  if (HAS_SMI AND CMAKE_CUDA_COMPILER_VERSION MATCHES "11\.[6-9]\.*")
    execute_process(
      COMMAND
      bash
      "-o"
      "pipefail"
      "-c"
      "nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | uniq"
      OUTPUT_VARIABLE SMI_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE STATUS
    )
    if (STATUS AND NOT STATUS EQUAL 0)
      message("Warning: failed to determine CUDA_ARCH automatically: ${SMI_OUTPUT}")
    else()
      STRING(REPLACE "." "" CUDA_ARCH ${SMI_OUTPUT})
      # Verify that only one compute capability was returned
      STRING(REGEX MATCHALL "\n" NEW_LINES "${CUDA_ARCH}")
      # NUM_COMPUTE_CAPS is off by one, as there will be one less newline than there are compute capabilities
      list(LENGTH NEW_LINES NUM_COMPUTE_CAPS)
      # If there are multiple compute capabilities, unset the CUDA_ARCH, fall back to the default below.
      if(NUM_COMPUTE_CAPS GREATER_EQUAL 1)
        unset(CUDA_ARCH)
      endif()
    endif()
  endif()
endif()
# If still not set, default to 75
if (NOT CUDA_ARCH)
  set(CUDA_ARCH "75")
endif()
message("Cuda Arch: " ${CUDA_ARCH})
set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH}")
unset(CUDA_ARCH CACHE)

target_link_libraries(${LIBRARY_NAME} PRIVATE -lcurand -lcuda -lcudart -lcudadevrt -lnvrtc)
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
