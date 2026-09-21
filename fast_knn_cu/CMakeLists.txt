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


# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/third_party/CMake"
  ${CMAKE_MODULE_PATH}
  )

# Add PyTorch
include_directories(${CONDA_PREFIX}/include)
include_directories(/usr/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)


find_package(Torch REQUIRED)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                     )



# Add cudaKDTree
add_subdirectory(third_party/cudaKDTree)
include_directories(third_party/cudaKDTree)
target_link_libraries(cudaKDTree)


string(REGEX REPLACE "^([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" TORCH_VERSION_CLEAN "${TORCH_VERSION}")
message(STATUS "Clean Torch version: ${TORCH_VERSION_CLEAN}")
if(${TORCH_VERSION_CLEAN} VERSION_GREATER "2.5.0")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  message(STATUS "Found Torch version: ${TORCH_VERSION}. CXX11ABI=1")
else()
  message(STATUS "Found Torch version: ${TORCH_VERSION}. CXX11ABI=0")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

pybind11_add_module(fast_cu_knn
  py_binding.cpp
  test_spatialkdtree.h
  test_spatialkdtree.cu
  torch_binding.h
  torch_binding.cu
)

# For CMake 3.16+
target_precompile_headers(fast_cu_knn PRIVATE <torch/extension.h>)

# py_binding imports special GPU-only code so it has to be compiled with NVCC
set_source_files_properties(py_binding.cpp PROPERTIES LANGUAGE CUDA)

# target_link_libraries(fast_cu_knn PRIVATE 
# ${CMAKE_CUDA_RUNTIME_LIBRARY} ${TORCH_PYTHON_LIBRARIES} ${TORCH_LIBRARIES} torch torch_cpu c10 torch_cuda m)
target_link_libraries(fast_cu_knn PRIVATE ${CMAKE_CUDA_RUNTIME_LIBRARY} ${TORCH_LIBRARIES})