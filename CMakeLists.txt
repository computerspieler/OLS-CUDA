cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
include(CTest)
 
add_library(ols STATIC
    gpu_timer.hpp
    matrix.cuh
    matrix.cu
    ols.cuh
    ols.cu
)

target_compile_features(ols PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(ols
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
 
add_executable(ols_test kernel.cu)
 
set_property(TARGET ols_test 
    PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(ols_test PRIVATE ols)
 
if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET ols_test 
        PROPERTY
        BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    )
endif()
