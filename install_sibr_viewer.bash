#!/bin/bash
# SIBR Viewer
cd SIBR_viewers
git checkout fossa_compatibility # needed to properly build it


rm -rf build # remove build directory to reset CMake Cache
# cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86"
cmake --build build -j24 --target install
cd ../..
