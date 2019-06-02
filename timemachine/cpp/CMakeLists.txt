cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(timemachine LANGUAGES CXX CUDA)

string(APPEND CMAKE_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61 -O2")

FIND_PACKAGE(CUDA QUIET)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set(CMAKE_INSTALL_PREFIX ${PARENT_DIR} CACHE PATH "Default Install Path" FORCE)
endif()




set(LIBRARY_NAME custom_ops.cpython-35m-x86_64-linux-gnu)
add_library(${LIBRARY_NAME} SHARED
  src/wrap_kernels.cpp
  src/context.cu
  src/optimizer.cu
  src/gpu_utils.cu
  src/langevin.cu
  src/potential.cu
  src/custom_nonbonded_gpu.cu
  src/custom_bonded_gpu.cu
)

include_directories(/home/yutong/venv/include/site/python3.5)
include_directories(/home/yutong/venv/include/python3.5m)
include_directories(/usr/local/cuda/include/)
include_directories(src/kernels)

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 11)
target_link_libraries(${LIBRARY_NAME} -lcublas -lcurand -lcudart)
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")


install(TARGETS ${LIBRARY_NAME} DESTINATION "lib")