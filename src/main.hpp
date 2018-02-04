#ifndef MAIN
#define MAIN
#include <array>
#include <climits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <google/dense_hash_map>

#include "src/externs.hpp"
#include "src/gpu.cuh"

// Dimensional coordinate + batch index
template <uint8_t D> using Coord = std::array<int64_t, D + 1>;

// Key for InOutMapping
// (pixeldistance, stride, kernelsize, dilation)
using InOutKey = std::array<int64_t, 4>;

template <typename T> std::size_t hash_vec(T p) {
  uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
  for (auto x : p) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    hash = hash ^ (x >> 31);
  }
  return hash;
}

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
  cublasCreate(&RETURN_VAR.cuhandle)

// Usage
// INITIALIZE_AND_REFERENCE(Metadata<Dim>, metadata, init_metadata)
//

template <uint8_t D>
long t_initialize_coords(const int64_t *coords, int64_t nrows,
                         int64_t pixel_dist, void **metadata);

template <uint8_t D>
long t_initialize_out_coords(int64_t pixel_dist, int64_t stride,
                             void **metadata);

template <uint8_t D>
long t_initialize_coords_with_duplicates(const int64_t *coords, int64_t nrows,
                                         int64_t pixel_dist, void **metadata);

template <uint8_t D>
long t_get_index_map(const int64_t *coords, int64_t nrows, int64_t *p_index_map,
                     int64_t index_map_nrows, int64_t pixel_dist,
                     void **metadata);

template <uint8_t D>
long t_get_num_coords(long pixel_dist, int64_t *nrows, void **metadata);

template <uint8_t D>
long t_get_coords(long *coords, int64_t pixel_dist, void **metadata);

template <uint8_t D> void t_clear(void **metadata);

template <uint8_t D>
long t_get_permutation(long *p_permutation, int64_t pixel_dist_src,
                       int64_t pixel_dist_dst, void **metadata);

template <uint8_t D>
long t_conv_fw(const float *p_in_feat, int64_t in_nchannel, float *p_out_feat,
               int64_t out_nchannel, const float *p_kernel, const float *p_bias,
               int64_t out_nrows, int64_t pixel_dist, int64_t stride,
               int64_t kernel_size, int64_t dilation, int64_t region_type,
               void **metadata);

template <uint8_t D>
long t_conv_bw(const float *p_in_feat, float *p_grad_in_feat,
               int64_t in_nchannel, float *p_grad_out_feat,
               int64_t out_nchannel, float *p_kernel, float *p_grad_kernel,
               float *p_grad_bias, int64_t out_nrows, int64_t pixel_dist,
               int64_t stride, int64_t kernel_size, int64_t dilation,
               void **metadata);

template <uint8_t D>
long t_conv_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                   float *d_out_feat, int64_t out_nchannel,
                   const float *d_kernel, const float *d_bias,
                   int64_t out_nrows, int64_t pixel_dist, int64_t stride,
                   int64_t kernel_size, int64_t dilation, int64_t region_type,
                   cudaStream_t stream, void **metadata);

template <uint8_t D>
long t_conv_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                   int64_t in_nchannel, float *d_grad_out_feat,
                   int64_t out_nchannel, float *d_kernel, float *d_grad_kernel,
                   float *d_grad_bias, int64_t out_nrows, int64_t pixel_dist,
                   int64_t stride, int64_t kernel_size, int64_t dilation,
                   cudaStream_t stream, void **metadata);
#endif
