#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <tuple>

#include "src/kernel_region.hpp"
#include "src/main.hpp"

#include "src/sparse_convolution.cuh"
#include "src/sparse_convolution.hpp"

#include "src/sparse_pooling.cuh"
#include "src/sparse_pooling.hpp"

/**
  Create <batch index + coordinate> to feature index mapping. The mapping will
  be used to create input index to output index mapping for convolution
  computation.
*/
template <uint8_t D>
CoordIndexMap<D> CreateCoordIndexMap(const int64_t *loc, int64_t nrows,
                                     int64_t ncols) {
  assert(ncols - 1 == D); // D+1 th coord is the batch index
  CoordIndexMap<D> coord_map;
  coord_map.map.resize(nrows);
  Coord<D> coord;
  for (int i = 0; i < nrows; i++) {
    std::copy(&loc[i * ncols], &loc[(i + 1) * ncols], coord.data());
    if (coord_map.map.find(coord) == coord_map.map.end()) {
      coord_map.map[coord] = i;
    } else {
      std::cout << "Duplicate key found. Use initialize_coords_with_duplicates "
                   "or remove duplicates"
                << std::endl;
      exit(-1);
    }
  }
  return coord_map;
}

/**
  Create <batch index + coordinate> to feature index mapping, but with
  duplicate check. The mapping will be used to create input index to output
  index mapping for convolution computation.
*/
template <uint8_t D>
CoordIndexMap<D> CreateDuplicateCoordIndexMap(const int64_t *loc, int64_t nrows,
                                              int64_t ncols) {
  assert(ncols - 1 == D); // D+1 th coord is the batch index
  int counter = 0;
  CoordIndexMap<D> coord_map;
  coord_map.map.resize(nrows);
  Coord<D> coord;
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
template <uint8_t D>
void CreateDuplicateIndexMap(const CoordIndexMap<D> coord_map,
                             const int64_t *loc, int64_t nrows,
                             int64_t *index_map, int64_t index_map_nrows) {
  int ncols = D + 1;
  Coord<D> coord;
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
*/
template <uint8_t D>
CoordIndexMap<D> CreateOutputCoordIndexMap(const CoordIndexMap<D> in_coord_map,
                                           int64_t pixel_dist, int64_t stride) {
  CoordIndexMap<D> out_coord_map;
  int new_pixel_dist = pixel_dist * stride;
  if (new_pixel_dist < 1) {
    throw std::invalid_argument("Invalid pixel distance");
  }
  if (stride > 1) {
    int n_out = 0;
    for (auto in_pair : in_coord_map.map) {
      Coord<D> coord(in_pair.first);
      for (int i = 0; i < D; i++) {
        coord[i] = int(coord[i] / new_pixel_dist) * new_pixel_dist;
      }
      if (out_coord_map.map.find(coord) == out_coord_map.map.end())
        out_coord_map.map[coord] = n_out++;
    }
  } else {
    out_coord_map = in_coord_map;
  }

  return out_coord_map;
}

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map}
*/
template <uint8_t D>
std::tuple<InOutMapPerKernel, InOutMapPerKernel>
CreateInOutPerKernel(const CoordIndexMap<D> in_coord_map,
                     const CoordIndexMap<D> out_coord_map, int64_t pixel_dist,
                     int64_t kernel_size, int64_t dilation, int64_t region_type,
                     int64_t *p_offset, int64_t n_offset) {
  int kernel_volume, kernel_ind = 0;
  if (region_type == 0) {
    kernel_volume = pow(kernel_size, D);
  } else if (region_type == 1) {
    kernel_volume = 2 * kernel_size * D + 1;
  } else if (region_type == 2) {
    kernel_volume = n_offset;
  } else {
    throw std::invalid_argument("Invalid region type");
  }

  InOutMapPerKernel in_map(kernel_volume), out_map(kernel_volume);
  for (auto const out_coord_iter : out_coord_map.map) {
    auto out_coord = out_coord_iter.first;
    auto kernel_region =
        KernelRegion<D>(out_coord, pixel_dist, kernel_size, dilation,
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

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map} for transposed convolution
*/
template <uint8_t D>
std::tuple<InOutMapPerKernel, InOutMapPerKernel> CreateInOutPerKernelTranspose(
    const CoordIndexMap<D> in_coord_map, const CoordIndexMap<D> out_coord_map,
    int64_t out_pixel_dist, int64_t kernel_size, int64_t dilation,
    int64_t region_type, int64_t *p_offset, int64_t n_offset) {
  int kernel_volume, kernel_ind = 0;
  if (region_type == 0) {
    kernel_volume = pow(kernel_size, D);
  } else if (region_type == 1) {
    kernel_volume = 2 * kernel_size * D + 1;
  } else if (region_type == 2) {
    kernel_volume = n_offset;
  } else {
    throw std::invalid_argument("Invalid region type");
  }

  InOutMapPerKernel in_map(kernel_volume), out_map(kernel_volume);
  for (auto const in_coord_iter : in_coord_map.map) {
    auto in_coord = in_coord_iter.first;
    auto kernel_region =
        KernelRegion<D>(in_coord, out_pixel_dist, kernel_size, dilation,
                        region_type, p_offset, n_offset);
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

/*
 * Given coordinates and the pixel distance, create index map
 */
template <uint8_t D>
long t_initialize_coords(const int64_t *coords, int64_t nrows,
                         int64_t pixel_dist, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  if (coord2inds->find(pixel_dist) != coord2inds->end()) {
    std::cout << "Key exists" << std::endl;
    return -1;
  }
  (*coord2inds)[pixel_dist] = CreateCoordIndexMap<D>(coords, nrows, D + 1);
}

/*
 * Given coordinates and the pixel distance, create index map and index map
 */
template <uint8_t D>
long t_initialize_coords_with_duplicates(const int64_t *coords, int64_t nrows,
                                         int64_t pixel_dist, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  if (coord2inds->find(pixel_dist) != coord2inds->end()) {
    std::cout << "Key exists" << std::endl;
    return -1;
  }

  (*coord2inds)[pixel_dist] =
      CreateDuplicateCoordIndexMap<D>(coords, nrows, D + 1);
}

/*
 * Given coordinates and the pixel distance, create index map and index map
 */
template <uint8_t D>
long t_get_index_map(const int64_t *coords, int64_t nrows, int64_t *p_index_map,
                     int64_t index_map_nrows, int64_t pixel_dist,
                     void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  if (coord2inds->find(pixel_dist) == coord2inds->end()) {
    std::cout << "Key doesn't exists" << std::endl;
    return -1;
  }
  CreateDuplicateIndexMap((*coord2inds)[pixel_dist], coords, nrows, p_index_map,
                          index_map_nrows);
}

/*
 * Create output map for a specific pixeldist, stride if coordmap[pixeldist]
 * exists.
 */
template <uint8_t D>
long t_initialize_out_coords(int64_t pixel_dist, int64_t stride,
                             bool is_transpose, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);
  int out_pixel_dist =
      is_transpose ? (pixel_dist / stride) : (pixel_dist * stride);
  if (out_pixel_dist < 1) {
    throw std::invalid_argument("Invalid pixel distance");
  }

  // Create index map and put it in the metadata
  auto coord2inds = &init_metadata.coord2inds;
  if (coord2inds->find(pixel_dist) == coord2inds->end()) {
    return -1;
  }

  if (stride > 1 && coord2inds->find(out_pixel_dist) == coord2inds->end()) {
    (*coord2inds)[out_pixel_dist] = CreateOutputCoordIndexMap<D>(
        (*coord2inds)[pixel_dist], pixel_dist, stride);
  }
  return 1;
}

template <uint8_t D>
long t_get_num_coords(int64_t pixel_dist, int64_t *p_nrows, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);
  if (init_metadata.coord2inds.find(pixel_dist) ==
      init_metadata.coord2inds.end()) {
    return -1;
  }
  *p_nrows = init_metadata.coord2inds[pixel_dist].map.size();
  return 1;
}

template <uint8_t D>
long t_get_coords(int64_t *p_coords, int64_t pixel_dist, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);
  auto coord2inds = &init_metadata.coord2inds;
  int nrows = 0, ncols = D + 1;
  if (coord2inds->find(pixel_dist) == coord2inds->end()) {
    return -1;
  }
  auto coord2ind = &(*coord2inds)[pixel_dist].map;
  nrows = coord2ind->size();
  if (nrows < 1)
    return -1;

  for (auto pair : *coord2ind)
    std::memcpy(&p_coords[ncols * pair.second], &pair.first,
                (D + 1) * sizeof(long));

  return 1;
}

/*
 * Given pixel_dist_src and pixel_dist_dst, find the respective coord_maps and
 * return the indices of the coord_map_ind in coord_map_dst
 */
template <uint8_t D>
long t_get_permutation(long *p_permutation, int64_t pixel_dist_src,
                       int64_t pixel_dist_dst, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);
  auto coord2inds = &init_metadata.coord2inds;
  int out_ind, in_ind;
  if (pixel_dist_src < pixel_dist_dst) {
    return -1;
  }
  if (coord2inds->find(pixel_dist_src) == coord2inds->end() ||
      coord2inds->find(pixel_dist_dst) == coord2inds->end()) {
    return -1;
  }
  auto coord2ind_src = &(*coord2inds)[pixel_dist_src].map;
  auto coord2ind_dst = &(*coord2inds)[pixel_dist_dst].map;

  int stride = pixel_dist_src / pixel_dist_dst;
  for (auto dst_pair : *coord2ind_dst) {
    out_ind = dst_pair.second;
    Coord<D> coord = dst_pair.first;
    for (int i = 0; i < D; i++) {
      coord[i] = (coord[i] / stride) * stride;
    }
    in_ind = (*coord2ind_src)[coord];
    p_permutation[out_ind] = in_ind;
  }
  return 1;
}

template <uint8_t D> void t_clear(void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);
  init_metadata.clear();
}

template <uint8_t D>
long t_conv_fw(const float *p_in_feat, int64_t in_nchannel, float *p_out_feat,
               int64_t out_nchannel, const float *p_kernel, int64_t out_nrows,
               int64_t pixel_dist, int64_t stride, int64_t kernel_size,
               int64_t dilation, int64_t region_type, int64_t *p_offset,
               int64_t n_offset, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  // Create output coordinate map if it doesn't exist
  if (stride > 1 &&
      p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    (*p_coord2inds)[pixel_dist * stride] = CreateOutputCoordIndexMap<D>(
        (*p_coord2inds)[pixel_dist], pixel_dist, stride);

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end()) {
    auto in_out_tuple = CreateInOutPerKernel<D>(
        (*p_coord2inds)[pixel_dist], (*p_coord2inds)[pixel_dist * stride],
        pixel_dist, kernel_size, dilation, region_type, p_offset, n_offset);
    (*p_in_maps)[key] = std::get<0>(in_out_tuple);
    (*p_out_maps)[key] = std::get<1>(in_out_tuple);
  }

  // Convolution
  SparseConvolutionForward<float>(p_in_feat, in_nchannel, p_out_feat,
                                  out_nchannel, p_kernel, (*p_in_maps)[key],
                                  (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D>
long t_conv_tr_fw(const float *p_in_feat, int64_t in_nchannel,
                  float *p_out_feat, int64_t out_nchannel,
                  const float *p_kernel, int64_t out_nrows, int64_t pixel_dist,
                  int64_t out_stride, int64_t kernel_size, int64_t dilation,
                  int64_t region_type, int64_t *p_offset, int64_t n_offset,
                  void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;
  int64_t out_pixel_dist;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end()) {
    std::cerr << "Input pixel distance does not exist";
    return -1;
  }

  // Create output coordinate map if it doesn't exist
  if (out_stride > 1 && pixel_dist % out_stride != 0) {
    std::cerr << "Current pixel dist is not divisibe by out_stride.\n";
    return -1;
  }

  // Set out pixel distance
  out_pixel_dist = pixel_dist / out_stride;

  if (out_stride > 1 &&
      p_coord2inds->find(out_pixel_dist) == p_coord2inds->end()) {
    std::cerr << "Coordinate map not defined for pixel dist.\n";
    return -1;
  }

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, out_stride, kernel_size, dilation, true};
  if (p_in_maps->find(key) == p_in_maps->end()) {
    auto in_out_tuple = CreateInOutPerKernelTranspose<D>(
        (*p_coord2inds)[pixel_dist], (*p_coord2inds)[out_pixel_dist],
        out_pixel_dist, kernel_size, dilation, region_type, p_offset, n_offset);
    (*p_in_maps)[key] = std::get<0>(in_out_tuple);
    (*p_out_maps)[key] = std::get<1>(in_out_tuple);
  }

  // Convolution
  SparseConvolutionForward<float>(p_in_feat, in_nchannel, p_out_feat,
                                  out_nchannel, p_kernel, (*p_in_maps)[key],
                                  (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D>
long t_conv_bw(const float *p_in_feat, float *p_grad_in_feat,
               int64_t in_nchannel, float *p_grad_out_feat,
               int64_t out_nchannel, float *p_kernel, float *p_grad_kernel,
               int64_t out_nrows, int64_t pixel_dist, int64_t stride,
               int64_t kernel_size, int64_t dilation, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  if (p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    return -1;

  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end())
    return -1;

  // Convolution
  SparseConvolutionBackward<float>(p_in_feat, p_grad_in_feat, in_nchannel,
                                   p_grad_out_feat, out_nchannel, p_kernel,
                                   p_grad_kernel, (*p_in_maps)[key],
                                   (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D>
long t_conv_tr_bw(const float *p_in_feat, float *p_grad_in_feat,
                  int64_t in_nchannel, float *p_grad_out_feat,
                  int64_t out_nchannel, float *p_kernel, float *p_grad_kernel,
                  int64_t out_nrows, int64_t pixel_dist, int64_t out_stride,
                  int64_t kernel_size, int64_t dilation, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  if (p_coord2inds->find(pixel_dist / out_stride) == p_coord2inds->end())
    return -1;

  InOutKey key = {pixel_dist, out_stride, kernel_size, dilation, true};
  if (p_in_maps->find(key) == p_in_maps->end())
    return -1;

  // Convolution
  SparseConvolutionBackward<float>(p_in_feat, p_grad_in_feat, in_nchannel,
                                   p_grad_out_feat, out_nchannel, p_kernel,
                                   p_grad_kernel, (*p_in_maps)[key],
                                   (*p_out_maps)[key], out_nrows);

  return 1;
}

template <uint8_t D>
long t_conv_fw_gpu(const float *p_in_feat, int64_t in_nchannel,
                   float *p_out_feat, int64_t out_nchannel,
                   const float *p_kernel, int64_t out_nrows, int64_t pixel_dist,
                   int64_t stride, int64_t kernel_size, int64_t dilation,
                   int64_t region_type, int64_t *p_offset, int64_t n_offset,
                   cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  // Create output coordinate map if it doesn't exist
  if (stride > 1 &&
      p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    (*p_coord2inds)[pixel_dist * stride] = CreateOutputCoordIndexMap<D>(
        (*p_coord2inds)[pixel_dist], pixel_dist, stride);

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end()) {
    auto in_out_tuple = CreateInOutPerKernel<D>(
        (*p_coord2inds)[pixel_dist], (*p_coord2inds)[pixel_dist * stride],
        pixel_dist, kernel_size, dilation, region_type, p_offset, n_offset);
    (*p_in_maps)[key] = std::get<0>(in_out_tuple);
    (*p_out_maps)[key] = std::get<1>(in_out_tuple);
  }

  // Convolution
  SparseConvolutionForwardGPU<float>(p_in_feat, in_nchannel, p_out_feat,
                                     out_nchannel, p_kernel, (*p_in_maps)[key],
                                     (*p_out_maps)[key], out_nrows,
                                     init_metadata.cuhandle, stream);

  return 1;
}

template <uint8_t D>
long t_conv_tr_fw_gpu(const float *p_in_feat, int64_t in_nchannel,
                      float *p_out_feat, int64_t out_nchannel,
                      const float *p_kernel, int64_t out_nrows,
                      int64_t pixel_dist, int64_t out_stride,
                      int64_t kernel_size, int64_t dilation,
                      int64_t region_type, int64_t *p_offset, int64_t n_offset,
                      cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;
  int64_t out_pixel_dist;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end()) {
    std::cerr << "Input pixel distance does not exist";
    return -1;
  }

  // Create output coordinate map if it doesn't exist
  if (out_stride > 1 && pixel_dist % out_stride != 0) {
    std::cerr << "Current pixel dist is not divisibe by out_stride.\n";
    return -1;
  }

  out_pixel_dist = pixel_dist / out_stride;

  // Create output coordinate map if it doesn't exist
  if (out_stride > 1 &&
      p_coord2inds->find(out_pixel_dist) == p_coord2inds->end()) {
    std::cerr << "Coordinate map not defined for pixel dist.\n";
    return -1;
  }

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, out_stride, kernel_size, dilation, true};
  if (p_in_maps->find(key) == p_in_maps->end()) {
    auto in_out_tuple = CreateInOutPerKernelTranspose<D>(
        (*p_coord2inds)[pixel_dist], (*p_coord2inds)[out_pixel_dist],
        out_pixel_dist, kernel_size, dilation, region_type, p_offset, n_offset);
    (*p_in_maps)[key] = std::get<0>(in_out_tuple);
    (*p_out_maps)[key] = std::get<1>(in_out_tuple);
  }

  // Convolution
  SparseConvolutionForwardGPU<float>(p_in_feat, in_nchannel, p_out_feat,
                                     out_nchannel, p_kernel, (*p_in_maps)[key],
                                     (*p_out_maps)[key], out_nrows,
                                     init_metadata.cuhandle, stream);

  return 1;
}

template <uint8_t D>
long t_conv_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                   int64_t in_nchannel, float *d_grad_out_feat,
                   int64_t out_nchannel, float *d_kernel, float *d_grad_kernel,
                   int64_t out_nrows, int64_t pixel_dist, int64_t stride,
                   int64_t kernel_size, int64_t dilation, cudaStream_t stream,
                   void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  if (p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    return -1;

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end())
    return -1;

  // Convolution
  SparseConvolutionBackwardGPU<float>(
      d_in_feat, d_grad_in_feat, in_nchannel, d_grad_out_feat, out_nchannel,
      d_kernel, d_grad_kernel, (*p_in_maps)[key], (*p_out_maps)[key], out_nrows,
      init_metadata.cuhandle, stream);

  return 1;
}

template <uint8_t D>
long t_conv_tr_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                      int64_t in_nchannel, float *d_grad_out_feat,
                      int64_t out_nchannel, float *d_kernel,
                      float *d_grad_kernel, int64_t out_nrows,
                      int64_t pixel_dist, int64_t out_stride,
                      int64_t kernel_size, int64_t dilation,
                      cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  if (p_coord2inds->find(pixel_dist / out_stride) == p_coord2inds->end())
    return -1;

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, out_stride, kernel_size, dilation, true};
  if (p_in_maps->find(key) == p_in_maps->end())
    return -1;

  // Convolution
  SparseConvolutionBackwardGPU<float>(
      d_in_feat, d_grad_in_feat, in_nchannel, d_grad_out_feat, out_nchannel,
      d_kernel, d_grad_kernel, (*p_in_maps)[key], (*p_out_maps)[key], out_nrows,
      init_metadata.cuhandle, stream);

  return 1;
}

template <uint8_t D>
long t_max_pooling_fw(const float *p_in_feat, float *p_out_feat,
                      int64_t *p_mask_index, int64_t nchannel,
                      int64_t out_nrows, int64_t pixel_dist, int64_t stride,
                      int64_t kernel_size, int64_t dilation,
                      int64_t region_type, int64_t *p_offset, int64_t n_offset,
                      void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  // Create output coordinate map if it doesn't exist
  if (stride > 1 &&
      p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    (*p_coord2inds)[pixel_dist * stride] = CreateOutputCoordIndexMap<D>(
        (*p_coord2inds)[pixel_dist], pixel_dist, stride);

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end()) {
    auto in_out_tuple = CreateInOutPerKernel<D>(
        (*p_coord2inds)[pixel_dist], (*p_coord2inds)[pixel_dist * stride],
        pixel_dist, kernel_size, dilation, region_type, p_offset, n_offset);
    (*p_in_maps)[key] = std::get<0>(in_out_tuple);
    (*p_out_maps)[key] = std::get<1>(in_out_tuple);
  }

  // Convolution
  SparseMaxPoolingForward<float>(p_in_feat, p_out_feat, p_mask_index, nchannel,
                                 (*p_in_maps)[key], (*p_out_maps)[key],
                                 out_nrows);

  return 1;
}

template <uint8_t D>
long t_max_pooling_bw(float *p_grad_in_feat, int64_t in_nrows,
                      float *p_grad_out_feat, int64_t out_nrows,
                      const int64_t *p_mask_index, int64_t nchannel,
                      int64_t pixel_dist, int64_t stride, int64_t kernel_size,
                      int64_t dilation, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  if (p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    return -1;

  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end())
    return -1;

  // Convolution
  SparseMaxPoolingBackward<float>(p_grad_in_feat, in_nrows, p_grad_out_feat,
                                  out_nrows, p_mask_index, nchannel,
                                  (*p_in_maps)[key], (*p_out_maps)[key]);

  return 1;
}

template <uint8_t D>
long t_max_pooling_fw_gpu(const float *d_in_feat, float *d_out_feat,
                          int64_t out_nrows, int64_t *d_mask_index,
                          int64_t nchannel, int64_t pixel_dist, int64_t stride,
                          int64_t kernel_size, int64_t dilation,
                          int64_t region_type, int64_t *p_offset,
                          int64_t n_offset, cudaStream_t stream,
                          void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  // Create output coordinate map if it doesn't exist
  if (stride > 1 &&
      p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    (*p_coord2inds)[pixel_dist * stride] = CreateOutputCoordIndexMap<D>(
        (*p_coord2inds)[pixel_dist], pixel_dist, stride);

  // Create in to out convolution mapping if it doesn't exist
  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end()) {
    auto in_out_tuple = CreateInOutPerKernel<D>(
        (*p_coord2inds)[pixel_dist], (*p_coord2inds)[pixel_dist * stride],
        pixel_dist, kernel_size, dilation, region_type, p_offset, n_offset);
    (*p_in_maps)[key] = std::get<0>(in_out_tuple);
    (*p_out_maps)[key] = std::get<1>(in_out_tuple);
  }

  // Convolution
  SparseMaxPoolingForwardGPU<float>(d_in_feat, d_out_feat, out_nrows,
                                    d_mask_index, nchannel, (*p_in_maps)[key],
                                    (*p_out_maps)[key], stream);

  return 1;
}

template <uint8_t D>
long t_max_pooling_bw_gpu(float *d_grad_in_feat, int64_t in_nrows,
                          float *d_grad_out_feat, int64_t out_nrows,
                          const int64_t *d_mask_index, int64_t nchannel,
                          int64_t pixel_dist, int64_t stride,
                          int64_t kernel_size, int64_t dilation,
                          cudaStream_t stream, void **metadata) {
  INITIALIZE_AND_REFERENCE(Metadata<D>, metadata, init_metadata);

  // Initialize all input, output coordinates, and convolution mapping
  auto p_coord2inds = &init_metadata.coord2inds;
  auto p_in_maps = &init_metadata.in_maps;
  auto p_out_maps = &init_metadata.out_maps;

  // Assume that the input coord2ind exists.
  if (p_coord2inds->find(pixel_dist) == p_coord2inds->end())
    return -1;

  if (p_coord2inds->find(pixel_dist * stride) == p_coord2inds->end())
    return -1;

  InOutKey key = {pixel_dist, stride, kernel_size, dilation, false};
  if (p_in_maps->find(key) == p_in_maps->end())
    return -1;

  // Convolution
  SparseMaxPoolingBackwardGPU<float>(d_grad_in_feat, in_nrows, d_grad_out_feat,
                                     out_nrows, d_mask_index, nchannel, stream);

  return 1;
}
