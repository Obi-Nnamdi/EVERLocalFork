cd submodules/cudaKDTree || return
rm -rf build
mkdir build
cd build || return
# When building directly we target a specific CUDA architecture.
CXX=$CXX CC=$CC cmake -DCMAKE_CUDA_ARCHITECTURES="50" ..
make -j8 || exit 1
cd ../../..