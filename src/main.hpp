#ifndef MAIN
#define MAIN
#include <array>
#include <climits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <google/dense_hash_map>

#include "src/externs.hpp"
#include "src/gpu.cuh"

// Dimensional coordinate + batch index
template <uint8_t D> using Coord = std::array<int64_t, D + 1>;

// Stride hash
template <uint8_t D> using Arr = std::array<uint64_t, D>;

template <typename T>
void print_arr(T arr) {
  for (auto i = arr.begin(); i != arr.end(); i++)
    std::cout << *i;
}

// Key for InOutMapping
// (pixel distance hash, stride hash, kernel size, dilation, is_transpose)
using InOutKey = std::array<uint64_t, 5>;

template <typename T> std::size_t hash_vec(T p) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (uint64_t x : p) {
    hash *= UINT64_C(1099511628211);
    hash ^= x;
  }
  return hash;
}

template <uint8_t D> struct ArrHash {
  std::size_t operator()(Arr<D> const &p) const { return hash_vec<Arr<D>>(p); }
};

struct InOutKeyHash {
  std::size_t operator()(InOutKey const &p) const {
    return hash_vec<InOutKey>(p);
  }
};

// Input index to output index mapping for each spatial kernel
using InOutMapPerKernel = std::vector<std::vector<int64_t>>;

// For Used for fast index of coordinate retrieval
template <uint8_t D> struct CoordHash {
  std::size_t operator()(Coord<D> const &p) const {
    return hash_vec<Coord<D>>(p);
  }
};

// Location to index of the feature
template <uint8_t D>
using _CoordIndexMap = google::dense_hash_map<Coord<D>, uint64_t, CoordHash<D>,
                                              std::equal_to<Coord<D>>>;

template <uint8_t D> class CoordIndexMap {
public:
  uint64_t
      ctr; // Count #active sites during output hash construction. Then store
           // offset within a batch.
  _CoordIndexMap<D> map;
  CoordIndexMap() : ctr(0) {
    // Sparsehash needs a key to be set aside and never used - we use
    // (Int_MAX,...,Int_MAX)
    Coord<D> empty_key;
    for (int i = 0; i < D + 1; ++i)
      empty_key[i] = LLONG_MAX;
    map.set_empty_key(empty_key);
  }
};

template <uint8_t D> class Metadata {
public:
  // Coordinate to index mapping for specific pixel distance.
  std::map<int64_t, CoordIndexMap<D>> coord2inds;
  // In to out index mapping for each kernel, pooling
  std::unordered_map<InOutKey, InOutMapPerKernel, InOutKeyHash> in_maps;
  std::unordered_map<InOutKey, InOutMapPerKernel, InOutKeyHash> out_maps;

  // cuBLAS handle
  cublasHandle_t cuhandle;

  Metadata() { cuhandle = NULL; }
  ~Metadata() { cublasDestroy(cuhandle); }
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
#define INITIALIZE_AND_REFERENCE(TYPE, VAR, RETURN_VAR)                        \
  if (VAR[0] == NULL)                                                          \
    VAR[0] = (void *)new TYPE;                                                 \
  TYPE &RETURN_VAR = *(TYPE *)VAR[0];                                          \
  if (RETURN_VAR.cuhandle == NULL)                                             \
    cublasCreate(&RETURN_VAR.cuhandle);

// Usage
// INITIALIZE_AND_REFERENCE(Metadata<Dim>, metadata, init_metadata)

// Macro for initialization
#define INITIALIZE_DEFAULT_VARS_AND_HASHES(IS_TRANSPOSE)                       \
  auto p_coord2inds = &init_metadata.coord2inds;                               \
  auto p_in_maps = &init_metadata.in_maps;                                     \
  auto p_out_maps = &init_metadata.out_maps;                                   \
                                                                               \
  auto pixel_dists = ToArray<D>(p_pixel_dist);                                 \
  auto strides = ToArray<D>(p_stride);                                         \
  auto kernel_size = ToArray<D>(p_kernel_size);                                \
  auto dilations = ToArray<D>(p_dilation);                                     \
                                                                               \
  auto pixel_dist_hash = hash_vec<Arr<D>>(pixel_dists);                        \
  auto stride_hash = hash_vec<Arr<D>>(strides);                                \
  auto kernel_size_hash = hash_vec<Arr<D>>(kernel_size);                       \
  auto dilation_hash = hash_vec<Arr<D>>(dilations);                            \
                                                                               \
  auto out_pixel_dists =                                                       \
      ComputeOutPixelDist<D>(pixel_dists, strides, IS_TRANSPOSE);              \
  auto out_pixel_dist_hash = hash_vec<Arr<D>>(out_pixel_dists);                \
                                                                               \
  InOutKey key = {pixel_dist_hash, stride_hash, kernel_size_hash,              \
                  dilation_hash, IS_TRANSPOSE};

// Macro for out map and kernel map initialization
#define INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(IS_TRANSPOSE)                     \
  if (p_coord2inds->find(pixel_dist_hash) == p_coord2inds->end()) {            \
    std::cerr << "The coord map doesn't exists for the given pixel dists"      \
              << std::endl;                                                    \
    return -1;                                                                 \
  }                                                                            \
                                                                               \
  if (IS_TRANSPOSE) {                                                          \
    if (p_coord2inds->find(out_pixel_dist_hash) == p_coord2inds->end()) {      \
      std::cerr << "Out coordinate map not defined for pixel dist.\n";         \
      return -1;                                                               \
    }                                                                          \
    if (p_in_maps->find(key) == p_in_maps->end()) {                            \
      auto in_out_tuple = CreateInOutPerKernelTranspose<D>(                    \
          (*p_coord2inds)[pixel_dist_hash],                                    \
          (*p_coord2inds)[out_pixel_dist_hash], out_pixel_dists, kernel_size,  \
          dilations, region_type, p_offset, n_offset);                         \
      (*p_in_maps)[key] = std::get<0>(in_out_tuple);                           \
      (*p_out_maps)[key] = std::get<1>(in_out_tuple);                          \
    }                                                                          \
  } else {                                                                     \
    if (p_coord2inds->find(out_pixel_dist_hash) == p_coord2inds->end())        \
      (*p_coord2inds)[out_pixel_dist_hash] = CreateOutputCoordIndexMap<D>(     \
          (*p_coord2inds)[pixel_dist_hash], pixel_dists, strides);             \
                                                                               \
    if (p_in_maps->find(key) == p_in_maps->end()) {                            \
      auto in_out_tuple = CreateInOutPerKernel<D>(                             \
          (*p_coord2inds)[pixel_dist_hash],                                    \
          (*p_coord2inds)[out_pixel_dist_hash], pixel_dists, kernel_size,      \
          dilations, region_type, p_offset, n_offset);                         \
      (*p_in_maps)[key] = std::get<0>(in_out_tuple);                           \
      (*p_out_maps)[key] = std::get<1>(in_out_tuple);                          \
    }                                                                          \
  }

// Checks for backward prop
#define BACKWARD_PROP_CHECK                                                    \
  if (p_coord2inds->find(pixel_dist_hash) == p_coord2inds->end())              \
    return -1;                                                                 \
                                                                               \
  if (p_coord2inds->find(out_pixel_dist_hash) == p_coord2inds->end())          \
    return -1;                                                                 \
                                                                               \
  if (p_in_maps->find(key) == p_in_maps->end())                                \
    return -1;

template <uint8_t D>
long t_initialize_coords(const int64_t *coords, int64_t nrows,
                         const int64_t *p_pixel_dist, void **metadata);

template <uint8_t D>
long t_initialize_out_coords(const int64_t *p_pixel_dist,
                             const int64_t *p_stride, bool is_transpose,
                             void **metadata);

template <uint8_t D>
long t_initialize_coords_with_duplicates(const int64_t *coords, int64_t nrows,
                                         const int64_t *p_pixel_dist,
                                         void **metadata);

template <uint8_t D>
long t_get_index_map(const int64_t *coords, int64_t nrows, int64_t *p_index_map,
                     int64_t index_map_nrows, const int64_t *p_pixel_dist,
                     void **metadata);

template <uint8_t D>
long t_get_num_coords(const int64_t *p_pixel_dist, int64_t *nrows,
                      void **metadata);

template <uint8_t D>
long t_get_coords(long *coords, const int64_t *p_pixel_dist, void **metadata);

template <uint8_t D> void t_clear(void **metadata);

template <uint8_t D>
long t_get_permutation(long *p_permutation, const int64_t *p_pixel_dist_src,
                       const int64_t *p_pixel_dist_dst, void **metadata);

template <uint8_t D>
long t_conv_fw(const float *p_in_feat, int64_t in_nchannel, float *p_out_feat,
               int64_t out_nchannel, const float *p_kernel, int64_t out_nrows,
               const int64_t *p_pixel_dist, const int64_t *p_stride,
               const int64_t *p_kernel_size, const int64_t *p_dilation,
               int64_t region_type, const int64_t *p_offset, int64_t n_offset,
               void **metadata);

template <uint8_t D>
long t_conv_tr_fw(const float *p_in_feat, int64_t in_nchannel,
                  float *p_out_feat, int64_t out_nchannel,
                  const float *p_kernel, int64_t out_nrows,
                  const int64_t *p_pixel_dist, const int64_t *p_stride,
                  const int64_t *p_kernel_size, const int64_t *p_dilation,
                  int64_t region_type, const int64_t *p_offset,
                  int64_t n_offset, void **metadata);

template <uint8_t D>
long t_conv_bw(const float *p_in_feat, float *p_grad_in_feat,
               int64_t in_nchannel, const float *p_grad_out_feat,
               int64_t out_nchannel, const float *p_kernel,
               float *p_grad_kernel, int64_t out_nrows,
               const int64_t *p_pixel_dist, const int64_t *p_stride,
               const int64_t *p_kernel_size, const int64_t *p_dilation,
               void **metadata);

template <uint8_t D>
long t_conv_tr_bw(const float *p_in_feat, float *p_grad_in_feat,
                  int64_t in_nchannel, const float *p_grad_out_feat,
                  int64_t out_nchannel, const float *p_kernel,
                  float *p_grad_kernel, int64_t out_nrows,
                  const int64_t *p_pixel_dist, const int64_t *p_stride,
                  const int64_t *p_kernel_size, const int64_t *p_dilation,
                  void **metadata);

template <uint8_t D>
long t_conv_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                   float *d_out_feat, int64_t out_nchannel,
                   const float *d_kernel, int64_t out_nrows,
                   const int64_t *p_pixel_dist, const int64_t *p_stride,
                   const int64_t *p_kernel_size, const int64_t *p_dilation,
                   int64_t region_type, const int64_t *p_offset,
                   int64_t n_offset, cudaStream_t stream, void **metadata);

template <uint8_t D>
long t_conv_tr_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                      float *d_out_feat, int64_t out_nchannel,
                      const float *d_kernel, int64_t out_nrows,
                      const int64_t *p_pixel_dist, const int64_t *p_stride,
                      const int64_t *p_kernel_size, const int64_t *p_dilation,
                      int64_t region_type, const int64_t *p_offset,
                      int64_t n_offset, cudaStream_t stream, void **metadata);

template <uint8_t D>
long t_conv_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                   int64_t in_nchannel, const float *d_grad_out_feat,
                   int64_t out_nchannel, const float *d_kernel,
                   float *d_grad_kernel, int64_t out_nrows,
                   const int64_t *p_pixel_dist, const int64_t *p_stride,
                   const int64_t *p_kernel_size, const int64_t *p_dilation,
                   cudaStream_t stream, void **metadata);

template <uint8_t D>
long t_conv_tr_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                      int64_t in_nchannel, const float *d_grad_out_feat,
                      int64_t out_nchannel, const float *d_kernel,
                      float *d_grad_kernel, int64_t out_nrows,
                      const int64_t *p_pixel_dist, const int64_t *p_stride,
                      const int64_t *p_kernel_size, const int64_t *p_dilation,
                      cudaStream_t stream, void **metadata);

template <uint8_t D>
long t_max_pooling_fw(const float *p_in_feat, float *p_out_feat,
                      int64_t *p_mask_index, int64_t nchannel,
                      int64_t out_nrows, const int64_t *p_pixel_dist,
                      const int64_t *p_stride, const int64_t *p_kernel_size,
                      const int64_t *p_dilation, int64_t region_type,
                      const int64_t *p_offset, int64_t n_offset,
                      void **metadata);

template <uint8_t D>
long t_max_pooling_bw(float *p_grad_in_feat, int64_t in_nrows,
                      float *p_grad_out_feat, int64_t out_nrows,
                      const int64_t *p_mask_index, int64_t nchannel,
                      const int64_t *p_pixel_dist, const int64_t *p_stride,
                      const int64_t *p_kernel_size, const int64_t *p_dilation,
                      void **metadata);

template <uint8_t D>
long t_max_pooling_fw_gpu(const float *d_in_feat, float *d_out_feat,
                          int64_t out_nrows, int64_t *d_mask_index,
                          int64_t nchannel, const int64_t *p_pixel_dist,
                          const int64_t *p_stride, const int64_t *p_kernel_size,
                          const int64_t *p_dilation, int64_t region_type,
                          const int64_t *p_offset, int64_t n_offset,
                          cudaStream_t stream, void **metadata);

template <uint8_t D>
long t_max_pooling_bw_gpu(float *d_grad_in_feat, int64_t in_nrows,
                          const float *d_grad_out_feat, int64_t out_nrows,
                          const int64_t *d_mask_index, int64_t nchannel,
                          const int64_t *p_pixel_dist, const int64_t *p_stride,
                          const int64_t *p_kernel_size,
                          const int64_t *p_dilation, cudaStream_t stream,
                          void **metadata);
#endif
