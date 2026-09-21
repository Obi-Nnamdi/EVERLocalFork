import torch
from build import fast_cu_knn


test_tensor = torch.rand((10_250_000, 3)).cuda()
test_tensor_2 = torch.rand((40_000_000, 3)).cuda()
radius = 2
# Fast!
print(fast_cu_knn.test_torch(test_tensor, test_tensor_2, radius))
