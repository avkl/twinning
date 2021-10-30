cmake_minimum_required(VERSION 3.4...3.18)
project(twinning_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

include_directories(nanoflann/include eigen)
add_subdirectory(pybind11)
pybind11_add_module(twinning_cpp src/twinning.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(twinning_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
