// Link C++ code to Python.
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "torch_binding.h"

PYBIND11_MODULE(fast_cu_knn, m) {
  m.doc() = "Python Bindings for cudaKDTree.";
  m.def("run_knn", runKnn);
}