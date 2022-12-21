cmake_minimum_required(VERSION 3.4...3.25)
set(CMAKE_CXX_STANDARD 11)

project(twinning_cpp)

include_directories(nanoflann/include)
add_subdirectory(pybind11)
pybind11_add_module(twinning_cpp twinning.cpp)

target_compile_definitions(twinning_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(twinning_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()