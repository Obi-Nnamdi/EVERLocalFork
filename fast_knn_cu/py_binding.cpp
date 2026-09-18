// Link C++ code to Python.
#include <cstddef>
#include <stdio.h>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(fast_cu_knn, m) {
    m.doc() = "pybind11 example module";

    // Add bindings here
    m.def("foo", []() {
        return "Hello, World!";
    });
}