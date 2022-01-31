cmake_minimum_required(VERSION 3.4...3.18)
project(twinning_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(nanoflann/include)
add_subdirectory(pybind11)
pybind11_add_module(twinning_cpp twinning.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(twinning_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(twinning_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()
