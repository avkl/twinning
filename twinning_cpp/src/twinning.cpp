#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nanoflann.hpp>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

std::vector<std::size_t> twin_cpp(pybind11::array_t<double> data) 
{
    auto data_access = data.unchecked<2>();
    std::cout << data_access(0, 3) << std::endl;

    return {1, 2, 3};
}

namespace py = pybind11;

PYBIND11_MODULE(twinning_cpp, m) {
    m.doc() = R"pbdoc(
        .. currentmodule:: twinning_cpp

        .. autosummary::
           :toctree: _generate

           twin_cpp
    )pbdoc";

    m.def("twin_cpp", &twin_cpp, R"pbdoc(
        Partition a dataset into statistically similar twin sets.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
