# CMake build for the HIP FP8 GEMM kernel + pybind11 surface.
#
# Driven from the repo root via `kernels/build.cmake` (see kernels/README.md
# for the layout).  This CMakeLists is self-contained for the proof-of-life
# kernel; the perf path is the Triton kernel in kernels/triton_kernels/.
#
# Build:
#   mkdir -p kernels/hip/fp8_attn/build && cd $_
#   cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
#   make -j

cmake_minimum_required(VERSION 3.21)
project(repercep_fp8_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CDNA3 / MI300X.  Required for the FP8 MFMA intrinsics in fp8_gemm.hip.
set(CMAKE_HIP_ARCHITECTURES "gfx942")

# Find torch (needed for the pybind layer).  We allow the user to skip the
# Python extension and just build the .a if torch isn't installed.
find_package(Torch QUIET)

set(SRC fp8_gemm.hip)

if(Torch_FOUND)
    list(APPEND SRC binding.cpp)
    add_library(repercep_fp8_hip SHARED ${SRC})
    target_link_libraries(repercep_fp8_hip PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(repercep_fp8_hip PRIVATE ${TORCH_INCLUDE_DIRS})
    # Python extensions on Linux are .so without the "lib" prefix.
    set_target_properties(repercep_fp8_hip PROPERTIES
        PREFIX ""
        SUFFIX ".so")
else()
    message(STATUS "Torch not found — building HIP kernel as a static library only")
    add_library(repercep_fp8_hip STATIC ${SRC})
endif()

target_compile_options(repercep_fp8_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math>)
