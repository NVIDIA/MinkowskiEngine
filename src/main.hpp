#ifndef MAIN
#define MAIN
#include <array>
#include <climits>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <google/dense_hash_map>

#include "src/externs.hpp"
#include "src/gpu.cuh"

// Dimensional coordinate + batch index
template <uint8_t D, typename Itype> using Coord = std::array<Itype, D + 1>;

// Stride hash
template <uint8_t D, typename Itype> using Arr = std::array<Itype, D>;

template <typename T> void print_arr(T arr) {
  for (auto i : arr)
    std::cout << i;
  std::cout << std::endl;
}

// Key for InOutMapping
// (in_coords_key, out_coords_key, stride hash, kernel size, dilation,
// is_transpose)
using InOutKey = std::array<uint64_t, 6>;

// uint64_t for unsigned long, must use CXX -m64
template <typename T> uint64_t hash_vec(T p) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (uint64_t x : p) {
    hash *= UINT64_C(1099511628211);
    hash ^= x;
  }
  return hash;
}

template <uint8_t D, typename Itype> struct ArrHash {
  uint64_t operator()(Arr<D, Itype> const &p) const {
    return hash_vec<Arr<D, Itype>>(p);
  }
};

struct InOutKeyHash {
  uint64_t operator()(InOutKey const &p) const { return hash_vec<InOutKey>(p); }
};

// Input index to output index mapping for each spatial kernel
template <typename Itype>
using InOutMapPerKernel = std::vector<std::vector<Itype>>;

// For Used for fast index of coordinate retrieval
template <uint8_t D, typename Itype> struct CoordHash {
  uint64_t operator()(Coord<D, Itype> const &p) const {
    return hash_vec<Coord<D, Itype>>(p);
  }
};

// Location to index of the feature
template <uint8_t D, typename Itype>
using _CoordIndexMap =
    google::dense_hash_map<Coord<D, Itype>, uint64_t, CoordHash<D, Itype>,
                           std::equal_to<Coord<D, Itype>>>;

template <uint8_t D, typename Itype> class CoordIndexMap {
public:
  _CoordIndexMap<D, Itype> map;
  CoordIndexMap() {
    Coord<D, Itype> empty_key;
    for (int i = 0; i < D + 1; ++i)
      empty_key[i] = -std::numeric_limits<Itype>::max();
    map.set_empty_key(empty_key);
  }
  size_t size() { return map.size(); }
};

template <uint8_t D, typename Itype> class Metadata {
public:
  // Coordinate to index mapping for specific pixel distance.
  std::map<uint64_t, CoordIndexMap<D, Itype>> coord2inds;
  // In to out index mapping for each kernel, pooling
  std::unordered_map<InOutKey, InOutMapPerKernel<Itype>, InOutKeyHash> in_maps;
  std::unordered_map<InOutKey, InOutMapPerKernel<Itype>, InOutKeyHash> out_maps;

  // cuBLAS handle
  cublasHandle_t cuhandle;
  cusparseHandle_t cushandle;

  Metadata() {
    cuhandle = NULL;
    cushandle = NULL;
  }
  ~Metadata() {
    cublasDestroy(cuhandle);
    cusparseDestroy(cushandle);
  }
  void clear() {
    coord2inds.clear();
    in_maps.clear();
    out_maps.clear();
  }
};

// Macro to initialize arguments passed as void*[1].
// The macro:
// - takes a pointer to a pointer [allocated as ffi.new('void *[1]')
// - if the pointer has not yet been initialized, create an object for it
// - initializes the cublas handle if not initialized
#define INITIALIZE_AND_REFERENCE(VAR, RETURN_VAR)                              \
  if (VAR[0] == NULL)                                                          \
    VAR[0] = (void *)new Metadata<D, Itype>;                                   \
  Metadata<D, Itype> &RETURN_VAR = *(Metadata<D, Itype> *)VAR[0];              \
  if (RETURN_VAR.cuhandle == NULL) {                                           \
    CUBLAS_CHECK(cublasCreate(&RETURN_VAR.cuhandle));                          \
    CUSPARSE_CHECK(cusparseCreate(&RETURN_VAR.cushandle));                     \
  }

// Usage
// INITIALIZE_AND_REFERENCE(Metadata<Dim, Itype>, metadata, init_metadata)

// Macro for initialization
#define INITIALIZE_DEFAULT_VARS_AND_HASHES(IS_TRANSPOSE)                       \
  auto p_coord2inds = &init_metadata.coord2inds;                               \
  auto p_in_maps = &init_metadata.in_maps;                                     \
  auto p_out_maps = &init_metadata.out_maps;                                   \
                                                                               \
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);                          \
  auto strides = ToArray<D, Itype>(p_stride);                                  \
  auto kernel_size = ToArray<D, Itype>(p_kernel_size);                         \
  auto dilations = ToArray<D, Itype>(p_dilation);                              \
                                                                               \
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);                 \
  auto stride_hash = hash_vec<Arr<D, Itype>>(strides);                         \
  auto kernel_size_hash = hash_vec<Arr<D, Itype>>(kernel_size);                \
  auto dilation_hash = hash_vec<Arr<D, Itype>>(dilations);                     \
                                                                               \
  auto out_pixel_dists =                                                       \
      ComputeOutPixelDist<D>(pixel_dists, strides, IS_TRANSPOSE);              \
  auto out_pixel_dist_hash = hash_vec<Arr<D, Itype>>(out_pixel_dists);

// Basic check
#define ASSERT_EQ(A, B)                                                        \
  if (A != B) {                                                                \
    std::cerr << "Assertion failed: " << #A << ": " << A << " != " << #B       \
              << ": " << B << " at " << __FILE__ << ":" << __LINE__            \
              << std::endl;                                                    \
    return -1;                                                                 \
  }

#define INITIALIZE_IN_COORDS_KEY                                               \
  /* Prioritize the p_in_coords_key */                                         \
  if (p_coord2inds->find(pixel_dist_hash) != p_coord2inds->end() &&            \
      *p_in_coords_key == 0) {                                                 \
    *p_in_coords_key = pixel_dist_hash;                                        \
  } else if (*p_in_coords_key > 0) {                                           \
    /* Check the validity of the key */                                        \
    if (p_coord2inds->find(*p_in_coords_key) == p_coord2inds->end()) {         \
      std::cerr << "Given in_coords_key is invalid" << std::endl               \
                << "in_coords_key: " << *p_in_coords_key << " at " << __FILE__ \
                << ":" << __LINE__ << std::endl;                               \
      return -1;                                                               \
    }                                                                          \
  } else {                                                                     \
    std::cerr << "The coord map doesn't exists for the given pixel dists"      \
              << " and in_coords_key." << std::endl                            \
              << "pixel_dist_hash: " << pixel_dist_hash << std::endl           \
              << "in_coords_key: " << *p_in_coords_key << " at " << __FILE__   \
              << ":" << __LINE__ << std::endl;                                 \
    return -1;                                                                 \
  }

#define INITIALIZE_OUT_COORDS_KEY                                              \
  /* Similarly, the p_out_coords_key takes priority */                         \
  /* If an out_coords does not exist, create one. */                           \
  if (p_coord2inds->find(out_pixel_dist_hash) == p_coord2inds->end() &&        \
      *p_out_coords_key == 0) {                                                \
    /* For valid convolution, implement a new function */                      \
    (*p_coord2inds)[out_pixel_dist_hash] =                                     \
        CreateOutputCoordIndexMap<D, Itype>((*p_coord2inds)[*p_in_coords_key], \
                                            pixel_dists, strides);             \
    *p_out_coords_key = out_pixel_dist_hash;                                   \
  } else if (p_coord2inds->find(out_pixel_dist_hash) != p_coord2inds->end() && \
             *p_out_coords_key == 0) {                                         \
    *p_out_coords_key = out_pixel_dist_hash;                                   \
  } else if (*p_out_coords_key > 0) {                                          \
    if (p_coord2inds->find(*p_out_coords_key) == p_coord2inds->end()) {        \
      std::cerr << "Given out_coords_key is invalid" << std::endl              \
                << "out_coords_key: " << *p_out_coords_key << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return -1;                                                               \
    }                                                                          \
  }

#define INITIALIZE_VALID_OUT_COORDS_KEY                                        \
  /* If an out_coords does not exist, create one. */                           \
  if (*p_out_coords_key == 0) {                                                \
    *p_out_coords_key = random();                                              \
    while (p_coord2inds->find(*p_out_coords_key) != p_coord2inds->end()) {     \
      *p_out_coords_key = random();                                            \
    }                                                                          \
    (*p_coord2inds)[*p_out_coords_key] =                                       \
        CreateValidOutputCoordIndexMap<D, Itype>(                              \
            (*p_coord2inds)[*p_in_coords_key], pixel_dists, kernel_size,       \
            dilations);                                                        \
  } else if (*p_out_coords_key > 0) {                                          \
    if (p_coord2inds->find(*p_out_coords_key) == p_coord2inds->end()) {        \
      std::cerr << "Given out_coords_key is invalid" << std::endl              \
                << "out_coords_key: " << *p_out_coords_key << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return -1;                                                               \
    }                                                                          \
  }

#define CREATE_KERNEL_MAP(KERNEL_MAP_KEY, IS_TRANSPOSE)                        \
  /* Create InOutKernelMap */                                                  \
  InOutKey KERNEL_MAP_KEY = {                                                  \
      *p_in_coords_key, *p_out_coords_key, stride_hash,                        \
      kernel_size_hash, dilation_hash,     IS_TRANSPOSE};                      \
  if (p_in_maps->find(KERNEL_MAP_KEY) == p_in_maps->end()) {                   \
    if (IS_TRANSPOSE) {                                                        \
      auto in_out = CreateInOutPerKernelTranspose<D, Itype>(                   \
          (*p_coord2inds)[pixel_dist_hash],                                    \
          (*p_coord2inds)[out_pixel_dist_hash], out_pixel_dists, kernel_size,  \
          dilations, region_type, p_offset, n_offset);                         \
      (*p_in_maps)[KERNEL_MAP_KEY] = std::move(std::get<0>(in_out));           \
      (*p_out_maps)[KERNEL_MAP_KEY] = std::move(std::get<1>(in_out));          \
    } else {                                                                   \
      auto in_out = CreateInOutPerKernel<D, Itype>(                            \
          (*p_coord2inds)[*p_in_coords_key],                                   \
          (*p_coord2inds)[*p_out_coords_key], pixel_dists, kernel_size,        \
          dilations, region_type, p_offset, n_offset);                         \
      (*p_in_maps)[KERNEL_MAP_KEY] = std::move(std::get<0>(in_out));           \
      (*p_out_maps)[KERNEL_MAP_KEY] = std::move(std::get<1>(in_out));          \
    }                                                                          \
  }

#define INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES                              \
  auto p_coord2inds = &init_metadata.coord2inds;                               \
  auto p_in_maps = &init_metadata.in_maps;                                     \
  auto p_out_maps = &init_metadata.out_maps;                                   \
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);                          \
  auto out_pixel_dists = Arr<D, Itype>();                                      \
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);                 \
  auto out_pixel_dist_hash = hash_vec<Arr<D, Itype>>(out_pixel_dists);         \
  auto stride_hash = hash_vec<Arr<D, Itype>>(Arr<D, Itype>());                 \
  auto kernel_size_hash = hash_vec<Arr<D, Itype>>(Arr<D, Itype>());            \
  auto dilation_hash = hash_vec<Arr<D, Itype>>(Arr<D, Itype>());

#define INITIALIZE_GLOBAL_OUT_COORDS_KEY                                       \
  /* Similarly, the p_out_coords_key takes priority */                         \
  /* If an out_coords does not exist, create one. */                           \
  if (p_coord2inds->find(out_pixel_dist_hash) == p_coord2inds->end() &&        \
      *p_out_coords_key == 0) {                                                \
    (*p_coord2inds)[out_pixel_dist_hash] =                                     \
        CreateOutputOriginCoordIndexMap<D, Itype>(                             \
            (*p_coord2inds)[*p_in_coords_key], 0);                             \
    *p_out_coords_key = out_pixel_dist_hash;                                   \
  } else if (p_coord2inds->find(out_pixel_dist_hash) != p_coord2inds->end() && \
             *p_out_coords_key == 0) {                                         \
    *p_out_coords_key = out_pixel_dist_hash;                                   \
  } else if (*p_out_coords_key > 0) {                                          \
    if (p_coord2inds->find(*p_out_coords_key) == p_coord2inds->end()) {        \
      std::cerr << "Given out_coords_key is invalid" << std::endl              \
                << "out_coords_key: " << *p_out_coords_key << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return -1;                                                               \
    }                                                                          \
  }

#define CREATE_GLOBAL_KERNEL_MAP(KERNEL_MAP_KEY, IS_TRANSPOSE)                 \
  /* Create InOutKernelMap */                                                  \
  InOutKey KERNEL_MAP_KEY = {                                                  \
      *p_in_coords_key, *p_out_coords_key, stride_hash,                        \
      kernel_size_hash, dilation_hash,     IS_TRANSPOSE};                      \
  if (p_in_maps->find(KERNEL_MAP_KEY) == p_in_maps->end()) {                   \
    auto in_out_tuple = CreateGlobalReductionInOutMap<D, Itype>(               \
        (*p_coord2inds)[*p_in_coords_key],                                     \
        (*p_coord2inds)[*p_out_coords_key]);                                   \
    (*p_in_maps)[KERNEL_MAP_KEY] = std::move(std::get<0>(in_out_tuple));       \
    (*p_out_maps)[KERNEL_MAP_KEY] = std::move(std::get<1>(in_out_tuple));      \
  }

#define BACKWARD_PROP_WITH_COORDS_KEYS_CHECK(KERNEL_MAP_KEY, IS_TRANSPOSE)     \
  /* Checks for backward pro */                                                \
  if (p_coord2inds->find(*p_in_coords_key) == p_coord2inds->end()) {           \
    std::cerr << "Given in_coords_key not found" << std::endl                  \
              << "in_coords_key: " << *p_in_coords_key << " at " << __FILE__   \
              << ":" << __LINE__ << std::endl;                                 \
    return -1;                                                                 \
  }                                                                            \
                                                                               \
  if (p_coord2inds->find(*p_out_coords_key) == p_coord2inds->end()) {          \
    std::cerr << "Given out_coords_key not found" << std::endl                 \
              << "out_coords_key: " << *p_out_coords_key << " at " << __FILE__ \
              << ":" << __LINE__ << std::endl;                                 \
    return -1;                                                                 \
  }                                                                            \
                                                                               \
  /* Create InOutKernelMap */                                                  \
  InOutKey KERNEL_MAP_KEY = {                                                  \
      *p_in_coords_key, *p_out_coords_key, stride_hash,                        \
      kernel_size_hash, dilation_hash,     IS_TRANSPOSE};                      \
                                                                               \
  if (p_in_maps->find(KERNEL_MAP_KEY) == p_in_maps->end()) {                   \
    std::cerr << "InOutKernelMap not found" << std::endl                       \
              << "#KERNEL_MAP_KEY hash: "                                      \
              << hash_vec<InOutKey>(KERNEL_MAP_KEY) << " at " << __FILE__      \
              << ":" << __LINE__ << std::endl;                                 \
    return -1;                                                                 \
  }

template <uint8_t D, typename Itype>
long t_initialize_coords(const Itype *coords, int nrows,
                         const Itype *p_pixel_dist, void **metadata);

template <uint8_t D, typename Itype>
long t_initialize_out_coords(uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key,
                             const Itype *p_pixel_dist, const Itype *p_stride,
                             bool is_transpose, void **metadata);

template <uint8_t D, typename Itype>
long t_initialize_valid_conv_out_coords(uint64_t *p_in_coords_key,
                                        uint64_t *p_out_coords_key,
                                        const Itype *p_pixel_dist,
                                        const Itype *p_stride,
                                        const Itype *p_kernel_size,
                                        const Itype *p_dilation,
                                        bool is_transpose, void **metadata);

template <uint8_t D, typename Itype>
long t_initialize_origin_coords(const uint64_t *p_in_coords_key,
                                const Itype *p_pixel_dist, int batch_size,
                                void **metadata);

template <uint8_t D, typename Itype>
long t_initialize_coords_with_duplicates(const Itype *coords, int nrows,
                                         const Itype *p_pixel_dist,
                                         void **metadata);

template <uint8_t D, typename Itype>
long t_get_index_map(const Itype *coords, int nrows, Itype *p_index_map,
                     int index_map_nrows, const Itype *p_pixel_dist,
                     void **metadata);

template <uint8_t D, typename Itype>
long t_get_num_coords(const uint64_t *p_coords_key, const Itype *p_pixel_dist,
                      int *nrows, void **metadata);

template <uint8_t D, typename Itype>
long t_get_coords(Itype *coords, const uint64_t *p_coords_key,
                  const Itype *p_pixel_dist, void **metadata);

template <uint8_t D, typename Itype> void t_clear(void **metadata);

template <uint8_t D, typename Itype>
long t_get_permutation(Itype *p_permutation, const Itype *p_pixel_dist_src,
                       const Itype *p_pixel_dist_dst, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_fw(const Dtype *p_in_feat, Itype in_nchannel, Dtype *p_out_feat,
               Itype out_nchannel, const Dtype *p_kernel, Itype out_nrows,
               const Itype *p_pixel_dist, const Itype *p_stride,
               const Itype *p_kernel_size, const Itype *p_dilation,
               Itype region_type, const Itype *p_offset, Itype n_offset,
               uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
               void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat, Itype in_nchannel,
               const Dtype *p_grad_out_feat, Itype out_nchannel,
               const Dtype *p_kernel, Dtype *p_grad_kernel, Itype out_nrows,
               const Itype *p_pixel_dist, const Itype *p_stride,
               const Itype *p_kernel_size, const Itype *p_dilation,
               uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
               void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_fw(const Dtype *p_in_feat, Itype in_nchannel, Dtype *p_out_feat,
                  Itype out_nchannel, const Dtype *p_kernel, Itype out_nrows,
                  const Itype *p_pixel_dist, const Itype *p_stride,
                  const Itype *p_kernel_size, const Itype *p_dilation,
                  Itype region_type, const Itype *p_offset, Itype n_offset,
                  uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                  void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                  Itype in_nchannel, const Dtype *p_grad_out_feat,
                  Itype out_nchannel, const Dtype *p_kernel,
                  Dtype *p_grad_kernel, Itype out_nrows,
                  const Itype *p_pixel_dist, const Itype *p_stride,
                  const Itype *p_kernel_size, const Itype *p_dilation,
                  uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                  void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_fw_gpu(const Dtype *d_in_feat, Itype in_nchannel, Dtype *d_out_feat,
                   Itype out_nchannel, const Dtype *d_kernel, Itype out_nrows,
                   const Itype *p_pixel_dist, const Itype *p_stride,
                   const Itype *p_kernel_size, const Itype *p_dilation,
                   Itype region_type, const Itype *p_offset, Itype n_offset,
                   uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                   cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_fw_gpu(const Dtype *d_in_feat, Itype in_nchannel,
                      Dtype *d_out_feat, Itype out_nchannel,
                      const Dtype *d_kernel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      Itype region_type, const Itype *p_offset, Itype n_offset,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                   Itype in_nchannel, const Dtype *d_grad_out_feat,
                   Itype out_nchannel, const Dtype *d_kernel,
                   Dtype *d_grad_kernel, Itype out_nrows,
                   const Itype *p_pixel_dist, const Itype *p_stride,
                   const Itype *p_kernel_size, const Itype *p_dilation,
                   uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                   cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                      Itype in_nchannel, const Dtype *d_grad_out_feat,
                      Itype out_nchannel, const Dtype *d_kernel,
                      Dtype *d_grad_kernel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                      Itype *p_mask_index, Itype nchannel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      Itype region_type, const Itype *p_offset, Itype n_offset,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                      Dtype *p_grad_out_feat, Itype out_nrows,
                      const Itype *p_mask_index, Itype nchannel,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_fw_gpu(const Dtype *d_in_feat, Dtype *d_out_feat,
                          Itype out_nrows, Itype *d_mask_index, Itype nchannel,
                          const Itype *p_pixel_dist, const Itype *p_stride,
                          const Itype *p_kernel_size, const Itype *p_dilation,
                          Itype region_type, const Itype *p_offset,
                          Itype n_offset, uint64_t *p_in_coords_key,
                          uint64_t *p_out_coords_key, cudaStream_t stream,
                          void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                          const Dtype *d_grad_out_feat, Itype out_nrows,
                          const Itype *d_mask_index, Itype nchannel,
                          const Itype *p_pixel_dist, const Itype *p_stride,
                          const Itype *p_kernel_size, const Itype *p_dilation,
                          uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                          cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                              Dtype *p_num_nonzero, Itype nchannel,
                              Itype out_nrows, const Itype *p_pixel_dist,
                              const Itype *p_stride, const Itype *p_kernel_size,
                              const Itype *p_dilation, Itype region_type,
                              const Itype *p_offset, Itype n_offset,
                              uint64_t *p_in_coords_key,
                              uint64_t *p_out_coords_key, Itype use_avg,
                              void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                              Dtype *p_grad_out_feat, Itype out_nrows,
                              const Dtype *p_num_nonzero, Itype nchannel,
                              const Itype *p_pixel_dist, const Itype *p_stride,
                              const Itype *p_kernel_size,
                              const Itype *p_dilation,
                              uint64_t *p_in_coords_key,
                              uint64_t *p_out_coords_key, Itype use_avg,
                              void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_fw_gpu(
    const Dtype *d_in_feat, Itype in_nrows, Dtype *d_out_feat, Itype out_nrows,
    Dtype *d_num_nonzero, Itype nchannel, const Itype *p_pixel_dist,
    const Itype *p_stride, const Itype *p_kernel_size, const Itype *p_dilation,
    Itype region_type, const Itype *p_offset, Itype n_offset,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, Itype use_avg,
    cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_bw_gpu(
    Dtype *d_grad_in_feat, Itype in_nrows, const Dtype *d_grad_out_feat,
    Itype out_nrows, const Dtype *d_num_nonzero, Itype nchannel,
    const Itype *p_pixel_dist, const Itype *p_stride,
    const Itype *p_kernel_size, const Itype *p_dilation,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, Itype use_avg,
    cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                    Dtype *p_num_nonzero, Itype nchannel, Itype out_nrows,
                    const Itype *p_pixel_dist, const Itype *p_stride,
                    const Itype *p_kernel_size, const Itype *p_dilation,
                    Itype region_type, const Itype *p_offset, Itype n_offset,
                    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                    void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                    Dtype *p_grad_out_feat, Itype out_nrows,
                    const Dtype *p_num_nonzero, Itype nchannel,
                    const Itype *p_pixel_dist, const Itype *p_stride,
                    const Itype *p_kernel_size, const Itype *p_dilation,
                    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                    void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_fw_gpu(const Dtype *d_in_feat, Itype in_nrows,
                        Dtype *d_out_feat, Itype out_nrows,
                        Dtype *d_num_nonzero, Itype nchannel,
                        const Itype *p_pixel_dist, const Itype *p_stride,
                        const Itype *p_kernel_size, const Itype *p_dilation,
                        Itype region_type, const Itype *p_offset,
                        Itype n_offset, uint64_t *p_in_coords_key,
                        uint64_t *p_out_coords_key, cudaStream_t stream,
                        void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                        const Dtype *d_grad_out_feat, Itype out_nrows,
                        const Dtype *d_num_nonzero, Itype nchannel,
                        const Itype *p_pixel_dist, const Itype *p_stride,
                        const Itype *p_kernel_size, const Itype *p_dilation,
                        uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                        cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                             Itype out_nrows, Itype nchannel,
                             Dtype *p_num_nonzero, const Itype *p_pixel_dist,
                             uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                             Dtype *p_grad_out_feat, Itype out_nrows,
                             Itype nchannel, const Dtype *p_num_nonzero,
                             const Itype *p_pixel_dist,
                             uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_fw_gpu(const Dtype *d_in_feat, Itype in_nrows,
                                 Dtype *d_out_feat, Itype out_nrows,
                                 Itype nchannel, Dtype *d_num_nonzero,
                                 const Itype *p_pixel_dist,
                                 uint64_t *p_in_coords_key,
                                 uint64_t *p_out_coords_key,
                                 cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                                 const Dtype *d_grad_out_feat, Itype out_nrows,
                                 Itype nchannel, const Dtype *d_num_nonzero,
                                 const Itype *p_pixel_dist,
                                 uint64_t *p_in_coords_key,
                                 uint64_t *p_out_coords_key,
                                 cudaStream_t stream, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_fw(const Dtype *p_in_feat, int in_nrows,
                           const Dtype *p_in_feat_global, int in_nrows_global,
                           Dtype *p_out_feat, int nchannel,
                           const Itype *p_pixel_dist, int op,
                           uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                           int in_nrows, const Dtype *p_in_feat_global,
                           Dtype *p_grad_in_feat_global, int in_nrows_global,
                           const Dtype *p_grad_out_feat, int nchannel,
                           const Itype *p_pixel_dist, int op,
                           uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_fw_gpu(const Dtype *d_in_feat, int in_nrows,
                               const Dtype *d_in_feat_global,
                               int in_nrows_global, Dtype *d_out_feat,
                               int nchannel, const Itype *p_pixel_dist, int op,
                               uint64_t *p_in_coords_key,
                               uint64_t *p_out_coords_key, cudaStream_t stream,
                               void **metadata);

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_bw_gpu(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nrows,
    const Dtype *d_in_feat_global, Dtype *d_grad_in_feat_global,
    int in_nrows_global, const Dtype *d_grad_out_feat, int nchannel,
    const Itype *p_pixel_dist, int op, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, void **metadata);
#endif
