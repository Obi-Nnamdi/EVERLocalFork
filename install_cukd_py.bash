cd fast_knn_cu || return

# If building with CMake directly
#######
# rm -rf build
# mkdir build
# cd build || return
# CXX=$CXX CC=$CC cmake -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86;90" ..
# make -j8 || exit 1

# Fast iteration
# CXX=$CXX CC=$CC cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES="60" -DTORCH_CUDA_ARCH_LIST="6.0" ..
# ninja -j10 || exit 1
#######

# Standard pip install (pulls from CMAKE_CUDA_ARCHITECTURES env variable)
pip install --no-build-isolation .
cd ../..