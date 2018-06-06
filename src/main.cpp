#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <tuple>

#include "src/kernel_region.hpp"
#include "src/main.hpp"

#include "src/sparse_convolution.cuh"
#include "src/sparse_convolution.hpp"

#include "src/sparse_pooling.cuh"
#include "src/sparse_pooling.hpp"

#include "src/sparse_broadcast.cuh"
#include "src/sparse_broadcast.hpp"

template <uint8_t D, typename Itype>
Arr<D, Itype> ComputeOutPixelDist(const Arr<D, Itype> pixel_dists,
                                  const Arr<D, Itype> strides,
                                  bool is_transpose) {
  Arr<D, Itype> out_pixel_dists;
  for (int i = 0; i < D; i++) {
    if (is_transpose)
      out_pixel_dists[i] = pixel_dists[i] / strides[i];
    else
      out_pixel_dists[i] = pixel_dists[i] * strides[i];

    if (out_pixel_dists[i] < 1)
      throw std::invalid_argument(
          "Out pixel distance is not a positive number");
  }
  return out_pixel_dists;
}

template <uint8_t D, typename Itype> Arr<D, Itype> ToArray(const Itype *ptr) {
  Arr<D, Itype> arr;
  std::copy(ptr, ptr + D, arr.begin());
  return arr;
}

/**
  Create <batch index + coordinate> to feature index mapping. The mapping will
  be used to create input index to output index mapping for convolution
  computation.
*/
template <uint8_t D, typename Itype>
CoordIndexMap<D, Itype> CreateCoordIndexMap(const Itype *loc, Itype nrows,
                                            Itype ncols) {
  assert(ncols - 1 == D); // D+1 th coord is the batch index
  CoordIndexMap<D, Itype> coord_map;
  coord_map.map.resize(nrows);
  Coord<D, Itype> coord;
  for (int i = 0; i < nrows; i++) {
    std::copy(&loc[i * ncols], &loc[(i + 1) * ncols], coord.data());
    if (coord_map.map.find(coord) == coord_map.map.end()) {
      coord_map.map[coord] = i;
    } else {
      std::cerr << "Duplicate key found. Use initialize_coords_with_duplicates "
                   "or remove duplicates"
                << std::endl;
    }
  }
  return coord_map;
}

/**
  Create <batch index + coordinate> to feature index mapping, but with
  duplicate check. The mapping will be used to create input index to output
  index mapping for convolution computation.
*/
template <uint8_t D, typename Itype>
CoordIndexMap<D, Itype> CreateDuplicateCoordIndexMap(const Itype *loc,
                                                     Itype nrows, Itype ncols) {
  assert(ncols - 1 == D); // D+1 th coord is the batch index
  int counter = 0;
  CoordIndexMap<D, Itype> coord_map;
  coord_map.map.resize(nrows);
  Coord<D, Itype> coord;
  for (int i = 0; i < nrows; i++) {
    std::copy(&loc[i * ncols], &loc[(i + 1) * ncols], coord.data());
    if (coord_map.map.find(coord) == coord_map.map.end()) {
      coord_map.map[coord] = counter++;
    }
  }
  return coord_map;
}

/**
 * Get coords index. Used to write index to given index_map pointer
 */
template <uint8_t D, typename Itype>
void CreateDuplicateIndexMap(CoordIndexMap<D, Itype> coord_map,
                             const Itype *loc, Itype nrows, Itype *index_map,
                             Itype index_map_nrows) {
  int ncols = D + 1;
  Coord<D, Itype> coord;
  for (int i = 0; i < nrows; i++) {
    std::copy(&loc[i * ncols], &loc[(i + 1) * ncols], coord.data());
    auto coord_iter = coord_map.map.find(coord);
    if (coord_iter == coord_map.map.end()) {
      index_map[i] = -1;
    } else {
      index_map[i] = coord_iter->second;
    }
  }
}

/**
  Given the input coordinate to index map, kernel size, stride, and dilation,
  compute the output coordinates and corresponding index.

  is_transpose is not used as we assume that the unpooled coords should
  correspond to one of the existing coord maps.
*/
template <uint8_t D, typename Itype>
CoordIndexMap<D, Itype>
CreateOutputCoordIndexMap(const CoordIndexMap<D, Itype> in_coord_map,
                          const Arr<D, Itype> pixel_dists,
                          const Arr<D, Itype> strides) {
  CoordIndexMap<D, Itype> out_coord_map;
  bool gt_one = false;
  for (auto s : strides) {
    if (s < 1)
      throw std::invalid_argument("Invalid pixel distance");
    if (s > 1)
      gt_one = true;
  }
  if (gt_one) {
    Arr<D, Itype> new_pixel_dists;
    int n_out = 0;
    for (int i = 0; i < D; i++)
      new_pixel_dists[i] = pixel_dists[i] * strides[i];
    for (auto in_pair : in_coord_map.map) {
      Coord<D, Itype> coord(in_pair.first);
      for (int j = 0; j < D; j++)
        coord[j] = int(coord[j] / new_pixel_dists[j]) * new_pixel_dists[j];
      if (out_coord_map.map.find(coord) == out_coord_map.map.end())
        out_coord_map.map[coord] = n_out++;
    }
  } else {
    out_coord_map = in_coord_map;
  }

  return out_coord_map;
}

/*
 * Coord map with the origin only
 */
template <uint8_t D, typename Itype>
CoordIndexMap<D, Itype>
CreateOutputOriginCoordIndexMap(const CoordIndexMap<D, Itype> in_coord_map,
                                Itype batch_size) {
  CoordIndexMap<D, Itype> out_coord_map;
  Coord<D, Itype> coord;
  int n_out = 0;
  if (batch_size < 1) {
    for (auto in_pair : in_coord_map.map) {
      Coord<D, Itype> coord(in_pair.first);
      for (int j = 0; j < D; j++)
        coord[j] = 0;
      if (out_coord_map.map.find(coord) == out_coord_map.map.end())
        out_coord_map.map[coord] = n_out++;
    }
  } else {
    for (int b = 0; b < batch_size; b++) {
      Coord<D, Itype> coord;
      for (int j = 0; j < D; j++)
        coord[j] = 0;
      coord[D] = b;
      out_coord_map.map[coord] = b;
    }
  }
  return out_coord_map;
}

template <uint8_t D, typename Itype>
long ComputeKernelVolume(Itype region_type, const Arr<D, Itype> kernel_size,
                         Itype n_offset) {
  int kernel_volume;
  if (region_type == 0) { // Hypercube
    kernel_volume = 1;
    for (auto k : kernel_size)
      kernel_volume *= k;
  } else if (region_type == 1) { // Hypercross
    kernel_volume = 1;
    for (auto k : kernel_size)
      kernel_volume += k - 1;
  } else if (region_type == 2) {
    kernel_volume = n_offset;
  } else {
    throw std::invalid_argument("Invalid region type");
  }
  return kernel_volume;
}

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map}
*/
template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CreateInOutPerKernel(const CoordIndexMap<D, Itype> in_coord_map,
                     const CoordIndexMap<D, Itype> out_coord_map,
                     const Arr<D, Itype> pixel_dists,
                     const Arr<D, Itype> kernel_size,
                     const Arr<D, Itype> dilations, Itype region_type,
                     const Itype *p_offset, Itype n_offset) {
  int kernel_volume, kernel_ind = 0;
  kernel_volume = ComputeKernelVolume<D>(region_type, kernel_size, n_offset);

  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);
  for (auto const out_coord_iter : out_coord_map.map) {
    auto out_coord = out_coord_iter.first;
    auto kernel_region =
        KernelRegion<D, Itype>(out_coord, pixel_dists, kernel_size, dilations,
                               region_type, p_offset, n_offset);
    kernel_ind = 0;
    for (auto point : kernel_region) {
      auto in_coord_iter = in_coord_map.map.find(point);
      if (in_coord_iter != in_coord_map.map.end()) {
        in_map[kernel_ind].push_back(in_coord_iter->second);
        out_map[kernel_ind].push_back(out_coord_iter.second);
      }
      kernel_ind++;
    }
  }
  return std::make_tuple(in_map, out_map);
}

template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CreateGlobalReductionInOutMap(const CoordIndexMap<D, Itype> in_coord_map,
                              const CoordIndexMap<D, Itype> out_coord_map) {
  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  std::map<Itype, Itype> in_out_map;
  // The out_coord_map.size() == 1
  for (auto const in_coord_iter : in_coord_map.map) {
    Coord<D, Itype> coord(in_coord_iter.first);
    for (int j = 0; j < D; j++)
      coord[j] = 0;
    auto out_coord_iter = out_coord_map.map.find(coord);
    if (out_coord_iter != out_coord_map.map.end()) {
      in_out_map[in_coord_iter.second] = out_coord_iter->second;
    } else {
      throw std::invalid_argument("Coord not found in out coord map\n");
    }
  }

  // Extract key value as in out (ascending) ordered by the in map
  for (auto const &pair : in_out_map) {
    in_map[0].push_back(pair.first);
    out_map[0].push_back(pair.second);
  }

  return std::make_tuple(in_map, out_map);
}

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map} for transposed convolution
*/
template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CreateInOutPerKernelTranspose(const CoordIndexMap<D, Itype> in_coord_map,
                              const CoordIndexMap<D, Itype> out_coord_map,
                              const Arr<D, Itype> out_pixel_dists,
                              const Arr<D, Itype> kernel_size,
                              const Arr<D, Itype> dilations, Itype region_type,
                              const Itype *p_offset, Itype n_offset) {
  int kernel_volume, kernel_ind = 0;
  kernel_volume = ComputeKernelVolume<D>(region_type, kernel_size, n_offset);

  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);
  for (auto const in_coord_iter : in_coord_map.map) {
    auto in_coord = in_coord_iter.first;
    auto kernel_region =
        KernelRegion<D, Itype>(in_coord, out_pixel_dists, kernel_size,
                               dilations, region_type, p_offset, n_offset);
    kernel_ind = 0;
    for (auto point : kernel_region) {
      auto out_coord_iter = out_coord_map.map.find(point);
      if (out_coord_iter != out_coord_map.map.end()) {
        in_map[kernel_ind].push_back(in_coord_iter.second);
        out_map[kernel_ind].push_back(out_coord_iter->second);
      }
      kernel_ind++;
    }
  }
  return std::make_tuple(in_map, out_map);
}

// For functions that communitate with C FFI, return -1 for failures rather
// than throwing errors which can't be handled in C FFI.

/*
 * Given coordinates and the pixel distance, create index map
 */
template <uint8_t D, typename Itype>
long t_initialize_coords(const Itype *coords, int nrows,
                         const Itype *p_pixel_dist, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (coord2inds->find(pixel_dist_hash) != coord2inds->end()) {
    std::cerr << "The coord map for the given pixel dists exists" << std::endl;
    return -1;
  }

  (*coord2inds)[pixel_dist_hash] = CreateCoordIndexMap<D>(coords, nrows, D + 1);
}

/*
 * Given coordinates and the pixel distance, create index map and index map
 */
template <uint8_t D, typename Itype>
long t_initialize_coords_with_duplicates(const Itype *coords, int nrows,
                                         const Itype *p_pixel_dist,
                                         void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (coord2inds->find(pixel_dist_hash) != coord2inds->end()) {
    std::cerr << "The coord map for the given pixel dists exists" << std::endl;
    return -1;
  }

  (*coord2inds)[pixel_dist_hash] =
      CreateDuplicateCoordIndexMap<D>(coords, nrows, D + 1);
}

/*
 * Given coordinates and the pixel distance, create index map and index map
 */
template <uint8_t D, typename Itype>
long t_get_index_map(const Itype *coords, int nrows, Itype *p_index_map,
                     int index_map_nrows, const Itype *p_pixel_dist,
                     void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (coord2inds->find(pixel_dist_hash) == coord2inds->end()) {
    std::cerr << "The coord map doesn't exists for the given pixel dists"
              << std::endl;
    return -1;
  }
  CreateDuplicateIndexMap<D>((*coord2inds)[pixel_dist_hash], coords, nrows,
                             p_index_map, index_map_nrows);
}

/*
 * Create output map for a specific pixeldist, stride if coordmap[pixeldist]
 * exists.
 */
template <uint8_t D, typename Itype>
long t_initialize_out_coords(const Itype *p_pixel_dist, const Itype *p_stride,
                             bool is_transpose, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)

  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  auto strides = ToArray<D, Itype>(p_stride);
  auto out_pixel_dists =
      ComputeOutPixelDist<D>(pixel_dists, strides, is_transpose);

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (coord2inds->find(pixel_dist_hash) == coord2inds->end()) {
    std::cerr << "Given input map for pixel dists does not exist";
    return -1;
  }

  auto out_pixel_dist_hash = hash_vec<Arr<D, Itype>>(out_pixel_dists);
  if (coord2inds->find(out_pixel_dist_hash) == coord2inds->end()) {
    if (is_transpose) {
      std::cerr << "The output coordinate map for transposed functions (e.g. "
                   "deconv) must be one of existing input coordinates"
                << std::endl;
      return -1;
    }
    (*coord2inds)[out_pixel_dist_hash] = CreateOutputCoordIndexMap<D, Itype>(
        (*coord2inds)[pixel_dist_hash], pixel_dists, strides);
  }
  return 1;
}

/*
 * Initialize origin map, if batch size is positive, use it for initialization.
 */
template <uint8_t D, typename Itype>
long t_initialize_origin_coords(const Itype *p_pixel_dist, int batch_size,
                                void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (coord2inds->find(pixel_dist_hash) == coord2inds->end()) {
    std::cerr << "The coord map for the given pixel dists does not exists"
              << std::endl;
    return -1;
  }

  // 0 initialized array for out pixel dist
  auto out_pixel_dist_hash = hash_vec<Arr<D, Itype>>(Arr<D, Itype>());
  (*coord2inds)[out_pixel_dist_hash] = CreateOutputOriginCoordIndexMap<D>(
      (*coord2inds)[pixel_dist_hash], batch_size);
}

template <uint8_t D, typename Itype>
long t_get_num_coords(const Itype *p_pixel_dist, int *p_nrows,
                      void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (init_metadata.coord2inds.find(pixel_dist_hash) ==
      init_metadata.coord2inds.end()) {
    return -1;
  }
  *p_nrows = init_metadata.coord2inds[pixel_dist_hash].map.size();
  return 1;
}

template <uint8_t D, typename Itype>
long t_get_coords(Itype *p_coords, const Itype *p_pixel_dist, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  auto coord2inds = &init_metadata.coord2inds;
  int nrows = 0, ncols = D + 1;

  auto pixel_dists = ToArray<D, Itype>(p_pixel_dist);
  auto pixel_dist_hash = hash_vec<Arr<D, Itype>>(pixel_dists);
  if (coord2inds->find(pixel_dist_hash) == coord2inds->end()) {
    return -1;
  }
  auto coord2ind = &(*coord2inds)[pixel_dist_hash].map;
  nrows = coord2ind->size();
  if (nrows < 1)
    return -1;

  // TODO Replace memcpy with copy
  for (auto pair : *coord2ind)
    std::memcpy(&p_coords[ncols * pair.second], &pair.first,
                (D + 1) * sizeof(Itype));

  return 1;
}

/*
 * Given pixel_dist_src and pixel_dist_dst, find the respective coord_maps and
 * return the indices of the coord_map_ind in coord_map_dst
 */
template <uint8_t D, typename Itype>
long t_get_permutation(Itype *p_permutation, const Itype *p_pixel_dist_src,
                       const Itype *p_pixel_dist_dst, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  auto coord2inds = &init_metadata.coord2inds;
  int out_ind, in_ind;
  auto pixel_dists_src = ToArray<D, Itype>(p_pixel_dist_src);
  auto pixel_dist_src_hash = hash_vec<Arr<D, Itype>>(pixel_dists_src);

  auto pixel_dists_dst = ToArray<D, Itype>(p_pixel_dist_dst);
  auto pixel_dist_dst_hash = hash_vec<Arr<D, Itype>>(pixel_dists_dst);

  auto strides = std::vector<Itype>(D);

  for (int i = 0; i < D; i++) {
    strides[i] = pixel_dists_src[i] / pixel_dists_dst[i];
    if (pixel_dists_src[i] < pixel_dists_dst[i]) {
      std::cerr << "Pixel dist src must be greater than pixel dist dst."
                << std::endl;
      return -1;
    }
  }

  if (coord2inds->find(pixel_dist_src_hash) == coord2inds->end() ||
      coord2inds->find(pixel_dist_dst_hash) == coord2inds->end()) {
    std::cerr << "Coordinate map for either pixel distances does not exist."
              << std::endl;
    return -1;
  }
  auto coord2ind_src = &(*coord2inds)[pixel_dist_src_hash].map;
  auto coord2ind_dst = &(*coord2inds)[pixel_dist_dst_hash].map;

  for (auto dst_pair : *coord2ind_dst) {
    out_ind = dst_pair.second;
    Coord<D, Itype> coord = dst_pair.first;
    for (int i = 0; i < D; i++) {
      coord[i] = (coord[i] / strides[i]) * strides[i];
    }
    in_ind = (*coord2ind_src)[coord];
    p_permutation[out_ind] = in_ind;
  }
  return 1;
}

template <uint8_t D, typename Itype> void t_clear(void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  init_metadata.clear();
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_fw(const Dtype *p_in_feat, Itype in_nchannel, Dtype *p_out_feat,
               Itype out_nchannel, const Dtype *p_kernel, Itype out_nrows,
               const Itype *p_pixel_dist, const Itype *p_stride,
               const Itype *p_kernel_size, const Itype *p_dilation,
               Itype region_type, const Itype *p_offset, Itype n_offset,
               void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(false)
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseConvolutionForward<Dtype, Itype>(
      p_in_feat, in_nchannel, p_out_feat, out_nchannel, p_kernel,
      (*p_in_maps)[key], (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_fw(const Dtype *p_in_feat, Itype in_nchannel, Dtype *p_out_feat,
                  Itype out_nchannel, const Dtype *p_kernel, Itype out_nrows,
                  const Itype *p_pixel_dist, const Itype *p_stride,
                  const Itype *p_kernel_size, const Itype *p_dilation,
                  Itype region_type, const Itype *p_offset, Itype n_offset,
                  void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(true)
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(true)
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseConvolutionForward<Dtype, Itype>(
      p_in_feat, in_nchannel, p_out_feat, out_nchannel, p_kernel,
      (*p_in_maps)[key], (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat, Itype in_nchannel,
               const Dtype *p_grad_out_feat, Itype out_nchannel,
               const Dtype *p_kernel, Dtype *p_grad_kernel, Itype out_nrows,
               const Itype *p_pixel_dist, const Itype *p_stride,
               const Itype *p_kernel_size, const Itype *p_dilation,
               void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  BACKWARD_PROP_CHECK

  SparseConvolutionBackward<Dtype, Itype>(
      p_in_feat, p_grad_in_feat, in_nchannel, p_grad_out_feat, out_nchannel,
      p_kernel, p_grad_kernel, (*p_in_maps)[key], (*p_out_maps)[key],
      out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                  Itype in_nchannel, const Dtype *p_grad_out_feat,
                  Itype out_nchannel, const Dtype *p_kernel,
                  Dtype *p_grad_kernel, Itype out_nrows,
                  const Itype *p_pixel_dist, const Itype *p_stride,
                  const Itype *p_kernel_size, const Itype *p_dilation,
                  void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(true)
  BACKWARD_PROP_CHECK

  SparseConvolutionBackward<Dtype, Itype>(
      p_in_feat, p_grad_in_feat, in_nchannel, p_grad_out_feat, out_nchannel,
      p_kernel, p_grad_kernel, (*p_in_maps)[key], (*p_out_maps)[key],
      out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_fw_gpu(const Dtype *d_in_feat, Itype in_nchannel, Dtype *d_out_feat,
                   Itype out_nchannel, const Dtype *d_kernel, Itype out_nrows,
                   const Itype *p_pixel_dist, const Itype *p_stride,
                   const Itype *p_kernel_size, const Itype *p_dilation,
                   Itype region_type, const Itype *p_offset, Itype n_offset,
                   cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false);
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(false);
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseConvolutionForwardGPU<Dtype, Itype>(
      d_in_feat, in_nchannel, d_out_feat, out_nchannel, d_kernel,
      (*p_in_maps)[key], (*p_out_maps)[key], out_nrows, init_metadata.cuhandle,
      stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_fw_gpu(const Dtype *d_in_feat, Itype in_nchannel,
                      Dtype *d_out_feat, Itype out_nchannel,
                      const Dtype *d_kernel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      Itype region_type, const Itype *p_offset, Itype n_offset,
                      cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(true)
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(true)
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseConvolutionForwardGPU<Dtype, Itype>(
      d_in_feat, in_nchannel, d_out_feat, out_nchannel, d_kernel,
      (*p_in_maps)[key], (*p_out_maps)[key], out_nrows, init_metadata.cuhandle,
      stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                   Itype in_nchannel, const Dtype *d_grad_out_feat,
                   Itype out_nchannel, const Dtype *d_kernel,
                   Dtype *d_grad_kernel, Itype out_nrows,
                   const Itype *p_pixel_dist, const Itype *p_stride,
                   const Itype *p_kernel_size, const Itype *p_dilation,
                   cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  BACKWARD_PROP_CHECK

  SparseConvolutionBackwardGPU<Dtype, Itype>(
      d_in_feat, d_grad_in_feat, in_nchannel, d_grad_out_feat, out_nchannel,
      d_kernel, d_grad_kernel, (*p_in_maps)[key], (*p_out_maps)[key], out_nrows,
      init_metadata.cuhandle, stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                      Itype in_nchannel, const Dtype *d_grad_out_feat,
                      Itype out_nchannel, const Dtype *d_kernel,
                      Dtype *d_grad_kernel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(true)
  BACKWARD_PROP_CHECK

  SparseConvolutionBackwardGPU<Dtype, Itype>(
      d_in_feat, d_grad_in_feat, in_nchannel, d_grad_out_feat, out_nchannel,
      d_kernel, d_grad_kernel, (*p_in_maps)[key], (*p_out_maps)[key], out_nrows,
      init_metadata.cuhandle, stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                      Itype *p_mask_index, Itype nchannel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      Itype region_type, const Itype *p_offset, Itype n_offset,
                      void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false);
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(false);
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseMaxPoolingForward<Dtype, Itype>(p_in_feat, p_out_feat, p_mask_index,
                                        nchannel, (*p_in_maps)[key],
                                        (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                      Dtype *p_grad_out_feat, Itype out_nrows,
                      const Itype *p_mask_index, Itype nchannel,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  BACKWARD_PROP_CHECK

  SparseMaxPoolingBackward<Dtype, Itype>(
      p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows, p_mask_index,
      nchannel, (*p_in_maps)[key], (*p_out_maps)[key]);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_fw_gpu(const Dtype *d_in_feat, Dtype *d_out_feat,
                          Itype out_nrows, Itype *d_mask_index, Itype nchannel,
                          const Itype *p_pixel_dist, const Itype *p_stride,
                          const Itype *p_kernel_size, const Itype *p_dilation,
                          Itype region_type, const Itype *p_offset,
                          Itype n_offset, cudaStream_t stream,
                          void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(false)
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseMaxPoolingForwardGPU<Dtype, Itype>(
      d_in_feat, d_out_feat, out_nrows, d_mask_index, nchannel,
      (*p_in_maps)[key], (*p_out_maps)[key], stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                          const Dtype *d_grad_out_feat, Itype out_nrows,
                          const Itype *d_mask_index, Itype nchannel,
                          const Itype *p_pixel_dist, const Itype *p_stride,
                          const Itype *p_kernel_size, const Itype *p_dilation,
                          cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  BACKWARD_PROP_CHECK

  SparseMaxPoolingBackwardGPU<Dtype, Itype>(d_grad_in_feat, in_nrows,
                                            d_grad_out_feat, out_nrows,
                                            d_mask_index, nchannel, stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                              Itype *p_num_nonzero, Itype nchannel,
                              Itype out_nrows, const Itype *p_pixel_dist,
                              const Itype *p_stride, const Itype *p_kernel_size,
                              const Itype *p_dilation, Itype region_type,
                              const Itype *p_offset, Itype n_offset,
                              void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false);
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(false);
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseNonzeroAvgPoolingForward<Dtype, Itype>(
      p_in_feat, p_out_feat, p_num_nonzero, nchannel, (*p_in_maps)[key],
      (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                              Dtype *p_grad_out_feat, Itype out_nrows,
                              const Itype *p_num_nonzero, Itype nchannel,
                              const Itype *p_pixel_dist, const Itype *p_stride,
                              const Itype *p_kernel_size,
                              const Itype *p_dilation, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  BACKWARD_PROP_CHECK

  SparseNonzeroAvgPoolingBackward<Dtype, Itype>(
      p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows, p_num_nonzero,
      nchannel, (*p_in_maps)[key], (*p_out_maps)[key]);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_fw_gpu(const Dtype *d_in_feat, Dtype *d_out_feat,
                                  Itype out_nrows, Itype *d_num_nonzero,
                                  Itype nchannel, const Itype *p_pixel_dist,
                                  const Itype *p_stride,
                                  const Itype *p_kernel_size,
                                  const Itype *p_dilation, Itype region_type,
                                  const Itype *p_offset, Itype n_offset,
                                  cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  INITIALIZE_OUT_COORDS_AND_KERNEL_MAP(false)
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseNonzeroAvgPoolingForwardGPU<Dtype, Itype>(
      d_in_feat, d_out_feat, out_nrows, d_num_nonzero, nchannel,
      (*p_in_maps)[key], (*p_out_maps)[key], stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                                  const Dtype *d_grad_out_feat, Itype out_nrows,
                                  const Itype *d_num_nonzero, Itype nchannel,
                                  const Itype *p_pixel_dist,
                                  const Itype *p_stride,
                                  const Itype *p_kernel_size,
                                  const Itype *p_dilation, cudaStream_t stream,
                                  void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_VARS_AND_HASHES(false)
  BACKWARD_PROP_CHECK

  SparseNonzeroAvgPoolingBackwardGPU<Dtype>(
      d_grad_in_feat, in_nrows, d_grad_out_feat, out_nrows, d_num_nonzero,
      nchannel, (*p_in_maps)[key], (*p_out_maps)[key], stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                             Itype out_nrows, Itype nchannel,
                             Itype *p_num_nonzero, const Itype *p_pixel_dist,
                             void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  INITIALIZE_GLOBAL_OUT_COORDS_AND_KERNEL_MAP
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseNonzeroAvgPoolingForward<Dtype, Itype>(
      p_in_feat, p_out_feat, p_num_nonzero, nchannel, (*p_in_maps)[key],
      (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                             Dtype *p_grad_out_feat, Itype out_nrows,
                             Itype nchannel, const Itype *p_num_nonzero,
                             const Itype *p_pixel_dist, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  BACKWARD_PROP_CHECK

  SparseNonzeroAvgPoolingBackward<Dtype, Itype>(
      p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows, p_num_nonzero,
      nchannel, (*p_in_maps)[key], (*p_out_maps)[key]);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_fw_gpu(const Dtype *d_in_feat, Dtype *d_out_feat,
                                 Itype out_nrows, Itype nchannel,
                                 Itype *d_num_nonzero,
                                 const Itype *p_pixel_dist, cudaStream_t stream,
                                 void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  INITIALIZE_GLOBAL_OUT_COORDS_AND_KERNEL_MAP
  ASSERT_EQ((*p_coord2inds)[out_pixel_dist_hash].size(), out_nrows)

  SparseNonzeroAvgPoolingForwardGPU<Dtype, Itype>(
      d_in_feat, d_out_feat, out_nrows, d_num_nonzero, nchannel,
      (*p_in_maps)[key], (*p_out_maps)[key], stream);

  return 1;
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                                 const Dtype *d_grad_out_feat, Itype out_nrows,
                                 Itype nchannel, const Itype *d_num_nonzero,
                                 const Itype *p_pixel_dist, cudaStream_t stream,
                                 void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  BACKWARD_PROP_CHECK

  SparseNonzeroAvgPoolingBackwardGPU<Dtype, Itype>(
      d_grad_in_feat, in_nrows, d_grad_out_feat, out_nrows, d_num_nonzero,
      nchannel, (*p_in_maps)[key], (*p_out_maps)[key], stream);

  return 1;
}

/*
 * Broadcast sum, multiplication operation.
 * First argument in_feat has in_nrows
 * output has in_nrows
 */
template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_fw(const Dtype *p_in_feat, int in_nrows,
                           const Dtype *p_in_feat_global, int in_nrows_global,
                           Dtype *p_out_feat, int nchannel,
                           const Itype *p_pixel_dist, int op, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  INITIALIZE_GLOBAL_OUT_COORDS_AND_KERNEL_MAP
  ASSERT_EQ((*p_coord2inds)[pixel_dist_hash].size(), in_nrows)

  SparseBroadcastForward<Dtype, Itype>(
      p_in_feat, in_nrows, p_in_feat_global, in_nrows_global, p_out_feat,
      nchannel, op, (*p_in_maps)[key], (*p_out_maps)[key]);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                           int in_nrows, const Dtype *p_in_feat_global,
                           Dtype *p_grad_in_feat_global, int in_nrows_global,
                           const Dtype *p_grad_out_feat, int nchannel,
                           const Itype *p_pixel_dist, int op, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  BACKWARD_PROP_CHECK

  SparseBroadcastBackward<Dtype, Itype>(
      p_in_feat, p_grad_in_feat, in_nrows, p_in_feat_global,
      p_grad_in_feat_global, in_nrows_global, p_grad_out_feat, nchannel, op,
      (*p_in_maps)[key], (*p_out_maps)[key]);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_fw_gpu(const Dtype *d_in_feat, int in_nrows,
                               const Dtype *d_in_feat_global,
                               int in_nrows_global, Dtype *d_out_feat,
                               int nchannel, const Itype *p_pixel_dist, int op,
                               cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  INITIALIZE_GLOBAL_OUT_COORDS_AND_KERNEL_MAP
  ASSERT_EQ((*p_coord2inds)[pixel_dist_hash].size(), in_nrows)

  SparseBroadcastForwardGPU<Dtype, Itype>(
      d_in_feat, in_nrows, d_in_feat_global, in_nrows_global, d_out_feat,
      nchannel, op, (*p_in_maps)[key], (*p_out_maps)[key],
      init_metadata.cushandle, stream);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                               int in_nrows, const Dtype *d_in_feat_global,
                               Dtype *d_grad_in_feat_global,
                               int in_nrows_global,
                               const Dtype *d_grad_out_feat, int nchannel,
                               const Itype *p_pixel_dist, int op,
                               cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(metadata, init_metadata)
  INITIALIZE_DEFAULT_GLOBAL_VARS_AND_HASHES
  BACKWARD_PROP_CHECK

  SparseBroadcastBackwardGPU<Dtype, Itype>(
      d_in_feat, d_grad_in_feat, in_nrows, d_in_feat_global,
      d_grad_in_feat_global, in_nrows_global, d_grad_out_feat, nchannel, op,
      (*p_in_maps)[key], (*p_out_maps)[key], init_metadata.cushandle, stream);
}
