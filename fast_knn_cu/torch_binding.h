#include <torch/extension.h>

torch::Tensor runKnn(const torch::Tensor& tree_points,
                     const torch::Tensor& query_points, const float radius);