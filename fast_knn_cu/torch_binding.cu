#include <cukd/builder.h>
#include <cukd/cuda_to_hip.h>
#include <cukd/knn.h>

#include "torch_binding.h"
// Only need the closest point for this kernel.
#define FIXED_K 1

// Float3 points in KD-Tree
using data_traits = cukd::default_data_traits<float3>;

// Special struct for storing float3 + original index in list using a torch
// tensor. From README of cuKD libary. x,y,z are position, and w stores a
// payload
struct PackedPointPlusPayload_traits
    : public cukd::default_data_traits<float3> {
  using point_t = float3;
  static inline __host__ __device__ const float3& get_point(
      const float4& packedPointAndPayload) {
    float3 point = make_float3(packedPointAndPayload.x, packedPointAndPayload.y,
                               packedPointAndPayload.z);
    return point;
  }

  static inline __host__ __device__ float get_coord(const float4& data,
                                                    int dim) {
    return cukd::get_coord(get_point(data), dim);
  }

  // Defined for completeness, since has_explicit_dim is false.
  static inline __host__ __device__ int get_dim(const float4&) { return -1; }
};

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

__global__ void KnnKernelStackFree(const float3* d_queries, int numQueries,
                                   float4* tree, int numTreePoints,
                                   const cukd::box_t<float3> world_bounds,
                                   int* d_results, int k, float radius) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries) return;

  cukd::FixedCandidateList<FIXED_K> result(
      radius);  // Fixed at 1, for generalization make template

  // Special case k = 1
  // int tree_point_ID = cukd::stackFree::fcp<float4,
  // PackedPointPlusPayload_traits>(
  //     d_queries[tid], tree, numTreePoints);

  cukd::stackFree::knn<decltype(result), float4, PackedPointPlusPayload_traits>(
      result, d_queries[tid], world_bounds, tree, numTreePoints);

  // -1 returned indices means no match
  for (int i = 0; i < k; i++) {
    int tree_point_ID = result.get_pointID(i);
    // Find the point and report its original index (bit casting float -> int)
    int ID = tree_point_ID < 0 ? -1 : __float_as_int(tree[tree_point_ID].w);
    d_results[tid * k + i] = ID;
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

torch::Tensor runKnnStackFree(const torch::Tensor& tree_points,
                              const torch::Tensor& query_points, const int K,
                              const float radius) {
  // Establish pre-condition (Contiguous CUDA tensor of N x 3)
  CHECK_FLOAT_DIM3(tree_points);
  CHECK_FLOAT_DIM3(query_points);
  CHECK(K > 0);
  CHECK(K <= FIXED_K);

  const int64_t numTreePoints = tree_points.size(0);
  const int64_t numQueryPoints = query_points.size(0);

  // Create an additional index dimension for the tree_points tensor
  // then bitcast it to a float32 (casting is undone in KnnKernelStackFree)
  auto int_opts = query_points.options().dtype(torch::kInt32);
  torch::Tensor index_tensor = torch::arange(numTreePoints, int_opts)
                                   .view({numTreePoints, 1})
                                   .view(torch::kFloat32);

  // Create a new tensor with the original points and indices together
  torch::Tensor points_and_index_tensor =
      torch::concat({tree_points, index_tensor}, /* dim = */ 1);

  float4* tree_pts_ptr =
      reinterpret_cast<float4*>(points_and_index_tensor.data_ptr());
  float3* query_points_ptr = reinterpret_cast<float3*>(query_points.data_ptr());

  // Build standard KD-Tree (reorders array)
  cukd::box_t<float3>* world_bounds;
  cudaMallocManaged(&world_bounds, sizeof(cukd::box_t<float3>));
  cukd::buildTree<float4, PackedPointPlusPayload_traits>(
      tree_pts_ptr, numTreePoints, world_bounds);

  CUKD_CUDA_SYNC_CHECK();

  // Create our results tensor on CUDA
  torch::Tensor results_indices =
      torch::full({numQueryPoints, K}, -1, int_opts);
  int* results_ptr = reinterpret_cast<int*>(results_indices.data_ptr());

  CUKD_CUDA_SYNC_CHECK();
  // Run our KNN Kernel
  int threadsPerBlock = 1024;
  int numBlocks = (numQueryPoints + threadsPerBlock - 1) / threadsPerBlock;

  KnnKernelStackFree<<<numBlocks, threadsPerBlock>>>(
      query_points_ptr, numQueryPoints, tree_pts_ptr, numTreePoints,
      *world_bounds, results_ptr, K, radius);
  cudaDeviceSynchronize();

  // Cleanup
  cudaFree(world_bounds);
  // Free unneeded allocated tensors
  // https://discuss.pytorch.org/t/how-to-manually-delete-free-a-tensor-in-aten/64153/3
  points_and_index_tensor = torch::Tensor();
  index_tensor = torch::Tensor();

  return results_indices;
}