cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA GPU architectures")
endif()

project(learn_cuda LANGUAGES CXX CUDA)

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

add_executable(learn_cuda
    src/main.cu
)

target_link_libraries(learn_cuda PRIVATE CUDA::cudart CUDA::cublas)

target_include_directories(learn_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/cutlass/include
)

target_compile_features(learn_cuda PRIVATE cxx_std_17 cuda_std_17)

target_compile_options(learn_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;--ptxas-options=-v>
)

set_target_properties(learn_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

# ---------------------------------------------------------------------------
# Auto-generate kernel resource usage report after every build
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_command(TARGET learn_cuda POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/scripts/gen_kernel_report.py
        --build-dir ${CMAKE_BINARY_DIR}
        --target learn_cuda
        --output ${CMAKE_SOURCE_DIR}/KERNEL_REPORT.md
    COMMENT "Generating CUDA kernel resource report (KERNEL_REPORT.md)..."
    VERBATIM
)
