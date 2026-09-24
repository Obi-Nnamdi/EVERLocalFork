# Gets copied to build directory using CMake.
import torch
from .fast_cu_knn import run_knn

test_tensor = torch.rand((10_250_000, 3)).cuda()
test_tensor_2 = torch.rand((40_000_000, 3)).cuda()
radius = 2
# Fast!
print(run_knn(test_tensor, test_tensor_2, radius))
