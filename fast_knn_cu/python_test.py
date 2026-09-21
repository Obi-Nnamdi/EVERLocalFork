import torch
from build import fast_cu_knn


print(fast_cu_knn.test_torch(torch.zeros((3, 1))))
