cmake_minimum_required(VERSION 3.20)
project(QuantumFlow LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
#   QF_ENABLE_CUDA   build the CUDA engine if an nvcc toolchain is found (ON)
#   QF_ENABLE_PYTHON build the pybind11 module qf_core (ON)
#   QF_NATIVE        optimise for this machine's CPU, incl. AVX2 (ON)
# ---------------------------------------------------------------------------
option(QF_ENABLE_CUDA "Build the CUDA Monte Carlo engine when nvcc is available" ON)
option(QF_ENABLE_PYTHON "Build the pybind11 module" ON)
option(QF_NATIVE "Compile with -march=native (enables AVX2 where the CPU has it)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(QF_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

find_package(OpenMP COMPONENTS CXX)

# ---------------------------------------------------------------------------
# Core header-only library (order book, analytic pricer, CPU Monte Carlo)
# ---------------------------------------------------------------------------
add_library(qf_headers INTERFACE)
target_include_directories(qf_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qf_headers INTERFACE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# CUDA engine (optional)
# ---------------------------------------------------------------------------
set(QF_HAVE_CUDA OFF)
if(QF_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # 61 = GTX 10xx, 75 = T4 / GTX 16xx / RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx
      set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    add_library(qf_cuda STATIC src/cuda/heston_mc.cu)
    target_link_libraries(qf_cuda PUBLIC qf_headers CUDA::cudart CUDA::curand)
    target_compile_options(qf_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
    target_compile_definitions(qf_cuda PUBLIC QF_WITH_CUDA)
    set(QF_HAVE_CUDA ON)
    message(STATUS "QuantumFlow: CUDA engine ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "QuantumFlow: no CUDA compiler found, building CPU-only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests (plain asserts, run with ctest)
# ---------------------------------------------------------------------------
enable_testing()
add_executable(test_orderbook tests/test_orderbook.cpp)
target_link_libraries(test_orderbook PRIVATE qf_headers)
add_test(NAME orderbook COMMAND test_orderbook)

add_executable(test_heston tests/test_heston.cpp)
target_link_libraries(test_heston PRIVATE qf_headers)
add_test(NAME heston COMMAND test_heston)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
add_executable(ob_latency_bench benchmarks/ob_latency_bench.cpp)
target_link_libraries(ob_latency_bench PRIVATE qf_headers)

add_executable(mc_cpu_bench benchmarks/mc_cpu_bench.cpp)
target_link_libraries(mc_cpu_bench PRIVATE qf_headers)

if(QF_HAVE_CUDA)
  add_executable(cuda_bench benchmarks/cuda_bench.cpp)
  target_link_libraries(cuda_bench PRIVATE qf_cuda)
endif()

# ---------------------------------------------------------------------------
# Python module qf_core (written next to the Python sources)
# ---------------------------------------------------------------------------
if(QF_ENABLE_PYTHON)
  find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(qf_core src/bridge/pybind_module.cpp)
  target_link_libraries(qf_core PRIVATE qf_headers)
  if(QF_HAVE_CUDA)
    target_link_libraries(qf_core PRIVATE qf_cuda)
  endif()
  set_target_properties(qf_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
endif()
