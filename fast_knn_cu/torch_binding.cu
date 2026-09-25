#include "torch_binding.h"
#include <cukd/cuda_to_hip.h>
#include <cukd/builder.h>
#include <cukd/knn.h>
// Only need the closest point for this kernel.
#define FIXED_K 1

// Float3 points in KD-Tree
using data_traits = cukd::default_data_traits<float3>;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) \
  TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) \
  CHECK_INPUT(x);           \
  CHECK_FLOAT(x);           \
  TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

// CUDA KNN Kernel (adapted from cudaKDTree/samples/knn-float3-spatialkdtree.cu)
__global__ void KnnKernel(const float3* d_queries, int numQueries,
                          const cukd::SpatialKDTree<float3, data_traits> tree,
                          int* d_results, int k, float radius) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries) return;

  cukd::FixedCandidateList<FIXED_K> result(
      radius);  // Fixed at 1, for generalization make template

  cukd::stackBased::knn<decltype(result), float3, data_traits>(result, tree,
                                                               d_queries[tid]);

  // -1 returned indices means no match
  for (int i = 0; i < k; i++) {
    int ID = result.get_pointID(i);
    d_results[tid * k + i] = ID < 0 ? -1 : ID;
  }
}

torch::Tensor runKnn(const torch::Tensor& tree_points,
                     const torch::Tensor& query_points, const int K,
                     const std::optional<float> radius) {
  // Establish pre-condition (Contiguous CUDA tensor of N x 3)
  CHECK_FLOAT_DIM3(tree_points);
  CHECK_FLOAT_DIM3(query_points);
  CHECK(K > 0);
  CHECK(K <= FIXED_K);

  const int64_t numTreePoints = tree_points.size(0);
  const int64_t numQueryPoints = query_points.size(0);

  float3* tree_pts_ptr = reinterpret_cast<float3*>(tree_points.data_ptr());
  float3* query_points_ptr = reinterpret_cast<float3*>(query_points.data_ptr());

  // Build Spatial KD-Tree (managed memory)
  cukd::SpatialKDTree<float3, data_traits> tree;
  cukd::BuildConfig buildConfig{};
  buildTree(tree, tree_pts_ptr, numTreePoints, buildConfig);

  CUKD_CUDA_SYNC_CHECK();

  // Create our results tensor on CUDA
  auto int_opts = query_points.options().dtype(torch::kInt32);
  torch::Tensor results_indices =
      torch::full({numQueryPoints, K}, -1, int_opts);
  int* results_ptr = reinterpret_cast<int*>(results_indices.data_ptr());

  float3 upperBounds = tree.bounds.upper;
  float3 lowerBounds = tree.bounds.upper;
  CUKD_CUDA_SYNC_CHECK();
  // Determine the maxRadius to query (bounding box len) so that we know for a
  // fact we'll never miss a KNN query
  float xDist = tree.bounds.upper.x - tree.bounds.lower.x;
  float yDist = tree.bounds.upper.y - tree.bounds.lower.y;
  float zDist = tree.bounds.upper.z - tree.bounds.lower.z;
  float maxRadius = sqrt(xDist * xDist + yDist * yDist + zDist * zDist) +
                    0.001;  // small bias value

  // Run our KNN Kernel (using custom radius if specified)
  int threadsPerBlock = 1024;
  int numBlocks = (numQueryPoints + threadsPerBlock - 1) / threadsPerBlock;
  KnnKernel<<<numBlocks, threadsPerBlock>>>(
      query_points_ptr, numQueryPoints, tree, results_ptr, K,
      radius.has_value() ? radius.value() : maxRadius);
  cudaDeviceSynchronize();

  // Clean up and get rid of our tree
  cukd::free(tree);

  return results_indices;
}