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

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

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
# target_link_libraries(cudaKDTree)


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
  torch_binding.h
  torch_binding.cu
)

# For CMake 3.16+
# TODO: Not sure how useful this is
target_precompile_headers(fast_cu_knn PRIVATE <torch/extension.h>)

# py_binding imports special GPU-only code so it has to be compiled with NVCC
set_source_files_properties(py_binding.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(fast_cu_knn PRIVATE ${CMAKE_CUDA_RUNTIME_LIBRARY} ${TORCH_LIBRARIES})

# Support pip building (Adapted from https://www.roma1.infn.it/~rovigatl/posts/pybind_cmake/)
# Create a staging directory for the Python package
set(PYTHON_BUILD_BASE_DIR "${CMAKE_BINARY_DIR}/python")
set(STAGING_PACKAGE_NAME "cukd_py")
set(FAST_CU_KNN_PACKAGE_DIR "${PYTHON_BUILD_BASE_DIR}/${STAGING_PACKAGE_NAME}")
# Make sure the directory exists
file(MAKE_DIRECTORY ${FAST_CU_KNN_PACKAGE_DIR})

# Place the compiled extension (.so) in that folder
set_target_properties(fast_cu_knn
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${FAST_CU_KNN_PACKAGE_DIR}"
    SUFFIX ".so"
)

# Move init.py to build directory
configure_file(__init__.py ${FAST_CU_KNN_PACKAGE_DIR}/__init__.py COPYONLY)

# Are we installing using pip + scikit-build?
if(SKBUILD)
  # install the package to a location where scikit-build can find it
  set(PIP_PACKAGE_NAME ${STAGING_PACKAGE_NAME})
  install(DIRECTORY ${FAST_CU_KNN_PACKAGE_DIR}/ DESTINATION ${PIP_PACKAGE_NAME})

  find_program(STUBGEN_PATH NAMES pybind11-stubgen)

  # Put our newly generated module on the pythnon path and generate stubs.
  if(STUBGEN_PATH)
      message(STATUS "Stubgen executable found at: ${STUBGEN_PATH}")
      # Add pip package to python path to make it readable by stubgen
      # set(ENV{PYTHONPATH} "${PYTHON_BUILD_BASE_DIR}")
      add_custom_command(
      TARGET fast_cu_knn POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHON_BUILD_BASE_DIR} ${STUBGEN_PATH} ${STAGING_PACKAGE_NAME} -o ${PYTHON_BUILD_BASE_DIR}
      COMMENT "generating python stubs"
      VERBATIM)
  else()
      message(WARNING "Stubgen Executable could not be found on the PATH. No typing stubs (.pyi files) will be generated.")
  endif()
endif()