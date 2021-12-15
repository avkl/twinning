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

    /*
        functions required by nanoflann
    */
    std::size_t kdtree_get_point_count() const
    {
        return data_access_->shape(0);
    }

    double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const 
    {
        return (*data_access_)(idx, dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const 
    { 
        return false; 
    }

    /*
        functions used while twinning  
    */
    const double* get_row(const std::size_t idx) const
    {
        return data_access_->data(idx, 0);
    }

    std::size_t nrow() const
    {
        return data_access_->shape(0);
    }

    std::size_t ncol() const
    {
        return data_access_->shape(1);
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

public:
    Twinning(py::array_t<double> data, std::size_t r, std::size_t u1, std::size_t leaf_size) : 
    r_(r), u1_(u1), leaf_size_(leaf_size)
    {
        data_ = std::make_shared<DF>(data);
    }

    std::vector<std::size_t> twin()
    {
        std::size_t N = data_->nrow();
        std::size_t dim = data_->ncol();

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
            tree.findNeighbors(resultSet, data_->get_row(position), nanoflann::SearchParams());
            indices.push_back(index[0]);
            
            for(std::size_t i = 0; i < r_; i++)
                tree.removePoint(index[i]);

            resultSet_next_u.init(&index_next_u, &distance_next_u);
            tree.findNeighbors(resultSet_next_u, data_->get_row(index[r_ - 1]), nanoflann::SearchParams());  
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
        std::size_t N = data_->nrow();
        std::size_t dim = data_->ncol();

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
                tree.findNeighbors(resultSet_f, data_->get_row(position), nanoflann::SearchParams());

                for(std::size_t i = 0; i < r_f; i++)
                    sequence.push_back(index_f[i]);

                break;
            }

            resultSet.init(index, distance);
            tree.findNeighbors(resultSet, data_->get_row(position), nanoflann::SearchParams());
            
            for(std::size_t i = 0; i < r_; i++)
            {
                sequence.push_back(index[i]);
                tree.removePoint(index[i]);
            }

            resultSet_next_u.init(&index_next_u, &distance_next_u);
            tree.findNeighbors(resultSet_next_u, data_->get_row(index[r_ - 1]), nanoflann::SearchParams());  
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


std::vector<std::size_t> multiplet_S3_cpp(py::array_t<double> data, std::size_t n, std::size_t u1, std::size_t leaf_size) 
{
    Twinning twinning(data, n, u1, leaf_size);
    return twinning.get_sequence();
}


PYBIND11_MODULE(twinning_cpp, m){
    m.doc() = R"pbdoc(
        .. currentmodule:: twinning_cpp

        .. autosummary::
           :toctree: _generate

           twin_cpp
           multiplet_S3_cpp
    )pbdoc";

    m.def("twin_cpp", &twin_cpp, R"pbdoc(
        Partition a dataset into statistically similar twin sets (C++ extension).
    )pbdoc");

    m.def("multiplet_S3_cpp", &multiplet_S3_cpp, R"pbdoc(
        Generate multiplets using strategy S3 (C++ extension).
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
