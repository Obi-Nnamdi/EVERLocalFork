#include <torch/extension.h>

/**
 * Points should be a N x 3 tensor on CUDA.
 * Uses radius dependent on tree_points bounds if not specified.
 */
torch::Tensor runKnn(const torch::Tensor& tree_points,
                     const torch::Tensor& query_points, const int K,
                     const std::optional<float> radius);