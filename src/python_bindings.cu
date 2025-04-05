#include <stdexcept>
#include <vector>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ols.cuh"

namespace py = pybind11;

void add_sample(OLS &ols, const py::array_t<float> sample, const float expected) {
    py::buffer_info buf_sample = sample.request();

    if(buf_sample.size != ols.dimension())
        throw std::range_error("Invalid size");

    ols.add_sample(static_cast<const float *>(buf_sample.ptr), expected);
}

py::array_t<float> retrieve_estimator(OLS &ols) {
    int dimension = ols.dimension();

    auto result = py::array_t<double>(dimension);
    py::buffer_info result_buf = result.request();

    float *result_ptr = static_cast<float *>(result_buf.ptr);

    std::vector<float> estimator = ols.retrieve_estimator();
    std::memcpy(result_ptr, estimator.data(), dimension * sizeof(float));

    return result;
}

PYBIND11_MODULE(ols, m) {
    py::class_<OLS>(m, "OLS")
        .def(py::init<const int>())
        .def("addSample", add_sample)
        .def("retrieveEstimator", retrieve_estimator);
}
