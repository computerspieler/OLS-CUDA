#include <pybind11/pybind11.h>

#include "ols.cuh"

namespace py = pybind11;

PYBIND11_MODULE(ols, m) {
    py::class_<OLS>(m, "OLS")
        .def(py::init<const int>());
}
