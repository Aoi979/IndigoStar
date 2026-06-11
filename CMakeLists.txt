cmake_minimum_required(VERSION 3.22.1)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA GPU architectures")
endif()

project(indigo_star LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Ensure Release means -O3 for both host and device code
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

foreach(cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(cuda_arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 80)
        message(FATAL_ERROR
            "This project uses cp.async and requires CUDA architecture >= 80. "
            "Got CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}.")
    endif()
endforeach()

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Determine which kernel sets to compile based on CUDA architectures
# ---------------------------------------------------------------------------
set(ENABLE_SM80_KERNELS 0)
set(ENABLE_SM90_KERNELS 0)
foreach(cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(cuda_arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 GREATER_EQUAL 80 AND CMAKE_MATCH_1 LESS 90)
        set(ENABLE_SM80_KERNELS 1)
    endif()
    if(cuda_arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 GREATER_EQUAL 90)
        set(ENABLE_SM90_KERNELS 1)
    endif()
endforeach()
message(STATUS "ENABLE_SM80_KERNELS=${ENABLE_SM80_KERNELS}")
message(STATUS "ENABLE_SM90_KERNELS=${ENABLE_SM90_KERNELS}")

add_executable(indigo_star
    src/main.cu
    src/core/cli.cu
    src/core/benchmark.cu
)

target_link_libraries(indigo_star PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver)

target_include_directories(indigo_star PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cutlass/include
    ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include
)

target_compile_definitions(indigo_star PRIVATE
    ENABLE_SM80_KERNELS=${ENABLE_SM80_KERNELS}
    ENABLE_SM90_KERNELS=${ENABLE_SM90_KERNELS}
)

target_compile_features(indigo_star PRIVATE cxx_std_17 cuda_std_17)

target_compile_options(indigo_star PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;--ptxas-options=-v>
)

set_target_properties(indigo_star PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

# ---------------------------------------------------------------------------
# Auto-generate kernel resource usage report after every build
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_command(TARGET indigo_star POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/scripts/gen_kernel_report.py
        --build-dir ${CMAKE_BINARY_DIR}
        --target indigo_star
        --output ${CMAKE_SOURCE_DIR}/KERNEL_REPORT.md
    COMMENT "Generating CUDA kernel resource report (KERNEL_REPORT.md)..."
    VERBATIM
)
