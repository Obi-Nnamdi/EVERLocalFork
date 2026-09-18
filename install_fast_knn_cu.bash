cd fast_knn_cu || return
rm -rf build
mkdir build
cd build || return
# When building directly we target a specific CUDA architecture.
CXX=$CXX CC=$CC cmake -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86;90" ..
make -j8 || exit 1
cd ../..