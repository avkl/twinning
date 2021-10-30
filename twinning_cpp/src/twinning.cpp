#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <nanoflann.hpp>
#include <omp.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> kdTree;

int add(int i, int j) 
{
    std::cout << "hello world!" << std::endl;
    std::cout << omp_get_max_threads() << std::endl;
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(twinning_cpp, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: twinning_cpp

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
