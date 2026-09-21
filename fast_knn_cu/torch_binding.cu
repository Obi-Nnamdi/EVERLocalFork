#include "torch_binding.h"
#include <cukd/cuda_to_hip.h>
#include <cukd/builder.h>
#include <cukd/knn.h>

torch::Tensor runKnn(const torch::Tensor& points) {
     torch::Tensor test_return = torch::zeros({3, 1});
     return test_return;
}
