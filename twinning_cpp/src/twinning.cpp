#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nanoflann.hpp>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

class DF
{
private:
    std::shared_ptr<py::detail::unchecked_reference<double, 2>> data_access_;

public:
    DF(py::array_t<double> data)
    {
        data_access_ = std::make_shared<py::detail::unchecked_reference<double, 2>>(data.unchecked<2>());
    }

    inline std::size_t kdtree_get_point_count() const
    {
        return data_access_->shape(0);
    }

    inline double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const 
    {
        return (*data_access_)(idx, dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const 
    { 
        return false; 
    }
};


typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF>, DF, -1, std::size_t> kdTree;


py::array_t<std::size_t> twin_cpp(py::array_t<double> data) 
{
    py::detail::unchecked_reference<double, 2> data_access = data.unchecked<2>();
    std::cout << data_access(0, 10) << std::endl;
    std::cout << data_access.shape(0) << std::endl;

    DF new_data(data);
    std::cout << new_data.kdtree_get_point_count() << std::endl;

    py::array_t<std::size_t> vec(5);
    return vec;
}


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
