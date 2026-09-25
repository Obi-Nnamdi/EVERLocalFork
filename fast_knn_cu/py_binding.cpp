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

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

PYBIND11_MODULE(fast_cu_knn, m) {
  m.doc() = "Python Bindings for cudaKDTree.";
  m.def("run_knn", runKnn,
        R"pbdoc(
Run cudaKD to generate a KD tree from tree_points and query it with query_points to find the K nearest neighbors.

Parameters
----------
tree_points : (T x 3) tensor
query_points : (Q x 3) tensor
k : int
radius : float (specified cut-off radius) or None (use bounds of the tree points as the max cutoff radius)

Returns
-------
result_indices : (Q x K) tensor
      )pbdoc",
        "tree_points"_a, "query_points"_a, "k"_a, "radius"_a = py::none());
}