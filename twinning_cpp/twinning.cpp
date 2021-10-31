#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nanoflann.hpp>
#include <vector>

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


typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF>, DF, -1, std::size_t> KDTree;


class Twinning
{
private:
    const std::size_t r_;
    const std::size_t u1_;
    const std::size_t leaf_size_;
    std::shared_ptr<DF> data_;
    std::shared_ptr<py::detail::unchecked_reference<double, 2>> data_access_;

public:
    Twinning(py::array_t<double> data, std::size_t r, std::size_t u1, std::size_t leaf_size) : 
    r_(r), u1_(u1), leaf_size_(leaf_size)
    {
        data_ = std::make_shared<DF>(data);
        data_access_ = std::make_shared<py::detail::unchecked_reference<double, 2>>(data.unchecked<2>());
    }

    std::vector<std::size_t> twin()
    {
        std::size_t N = data_access_->shape(0);
        std::size_t dim = data_access_->shape(1);

        KDTree tree(dim, *data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
        
        nanoflann::KNNResultSet<double> resultSet(r_);
        std::size_t index[r_];
        double distance[r_];

        nanoflann::KNNResultSet<double> resultSet_next_u(1);
        std::size_t index_next_u;
        double distance_next_u;

        std::vector<std::size_t> indices;
        indices.reserve(N / r_ + 1);
        std::size_t position = u1_;
        
        while(true)
        {
            resultSet.init(index, distance);
            tree.findNeighbors(resultSet, data_access_->data(position, 0), nanoflann::SearchParams());
            indices.push_back(index[0]);
            
            for(std::size_t i = 0; i < r_; i++)
                tree.removePoint(index[i]);

            resultSet_next_u.init(&index_next_u, &distance_next_u);
            tree.findNeighbors(resultSet_next_u, data_access_->data(index[r_ - 1], 0), nanoflann::SearchParams());  
            position = index_next_u;

            if(N - indices.size() * r_ <= r_)
            {
                indices.push_back(position);
                break;
            }
        }

        return indices;
    }

    std::vector<std::size_t> get_sequence()
    {
        std::size_t N = data_access_->shape(0);
        std::size_t dim = data_access_->shape(1);

        KDTree tree(dim, *data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
        
        nanoflann::KNNResultSet<double> resultSet(r_);
        std::size_t index[r_];
        double distance[r_];

        nanoflann::KNNResultSet<double> resultSet_next_u(1);
        std::size_t index_next_u;
        double distance_next_u;

        std::vector<std::size_t> sequence;
        sequence.reserve(N);
        std::size_t position = u1_;
        
        while(sequence.size() != N)
        {
            if(sequence.size() > N - r_)
                {
                    std::size_t r_f = N - sequence.size();
                    nanoflann::KNNResultSet<double> resultSet_f(r_f);
                    std::size_t index_f[r_f];
                    double distance_f[r_f];

                    resultSet_f.init(index_f, distance_f);
                    tree.findNeighbors(resultSet_f, data_access_->data(position, 0), nanoflann::SearchParams());

                    for(std::size_t i = 0; i < r_f; i++)
                        sequence.push_back(index_f[i]);

                    break;
                }

            resultSet.init(index, distance);
            tree.findNeighbors(resultSet, data_access_->data(position, 0), nanoflann::SearchParams());
            
            for(std::size_t i = 0; i < r_; i++)
            {
                sequence.push_back(index[i]);
                tree.removePoint(index[i]);
            }

            resultSet_next_u.init(&index_next_u, &distance_next_u);
            tree.findNeighbors(resultSet_next_u, data_access_->data(index[r_ - 1], 0), nanoflann::SearchParams());  
            position = index_next_u;
        }

        return sequence;
    }
};


std::vector<std::size_t> twin_cpp(py::array_t<double> data, std::size_t r, std::size_t u1, std::size_t leaf_size) 
{
    Twinning twinning(data, r, u1, leaf_size);
    return twinning.twin();
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
