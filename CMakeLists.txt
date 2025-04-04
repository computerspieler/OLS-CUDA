cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(ols
    MODULE src/matrix.cu
    MODULE src/ols.cu
    MODULE src/python_bindings.cu
)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(ols
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET ols 
        PROPERTY
        BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    )
endif()

target_link_libraries(ols PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
pybind11_extension(ols)
set_target_properties(ols
    PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
)

install(TARGETS ols DESTINATION .)
