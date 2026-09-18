# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on VERSION_GREATER_EQUAL which requires CMake v. 3.7.
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

# This sets up the name of our project.  For our purposes the main thing this controls is
# the name of the VS solution file.
project(fast_cu_knn_torch VERSION 1.0)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)


enable_language(CUDA)

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1  # Change this to the desired version
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(fast_cu_knn
  py_binding.cpp
)