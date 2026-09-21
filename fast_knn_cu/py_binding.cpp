// Link C++ code to Python.
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <test_spatialkdtree.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

PYBIND11_MODULE(fast_cu_knn, m) {
    m.doc() = "pybind11 example module";

    // Add bindings here
    m.def("foo", []() {
        return "Hello, World!";
    });

    m.def("run_main", run_main);
}