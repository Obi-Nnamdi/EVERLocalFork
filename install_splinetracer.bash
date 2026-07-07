#!/bin/bash

# Only needed with a clean install
# cp ever/new_files/*.py .
# cp -r ever/new_files/notebooks .
# cp ever/new_files/scene/* scene/
# cp ever/new_files/gaussian_renderer/* gaussian_renderer/
# cp ever/new_files/utils/* utils/

# git apply ../ever/new_files/sibr_patch.patch

# Build splinetracer
mkdir ever/build
cd ever/build
# CXX=/usr/bin/g++-11 CC=/usr/bin/gcc-11 cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -D_GLIBCXX_USE_CXX11_ABI=1 ..
# CXX=$CXX CC=$CC cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR ..
# CXX=$CXX CC=$CC cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
CXX=$CXX CC=$CC cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86;90" ..
make -j8 || exit 1
cd ../..