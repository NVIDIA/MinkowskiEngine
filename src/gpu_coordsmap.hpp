/* Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef GPU_COORDSMAP
#define GPU_COORDSMAP

#include <cmath>
#include <memory>
#include <set>
#include <tuple>
#include <torch/extension.h>

#include "3rdparty/gpu_coords_map/include/cuda_unordered_map.h"
#include "3rdparty/gpu_coords_map/include/coordinate.h"

#include "region.hpp"
#include "types.hpp"

namespace minkowski {

using std::reference_wrapper;
using std::set;
using std::tuple;
using std::vector;
using std::shared_ptr;

// TODO(ljm): enumerate and `DISPATCH` all possible combination
// D = 3
using CoordsToIndexMap_int_4_int_5_0 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 0>;

using CoordsToIndexMap_int_4_int_5_1 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 1>;

using CoordsToIndexMap_int_4_int_5_2 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 2>;

using CoordsToIndexMap_int_4_int_5_3 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 3>;

using CoordsToIndexMap_int_4_int_5_4 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 4>;

using CoordsToIndexMap_int_4_int_5_5 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 5>;

using CoordsToIndexMap_int_4_int_5_6 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 6>;

using CoordsToIndexMap_int_4_int_5_7 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 7>;

using CoordsToIndexMap_int_4_int_5_8 =
    cuda::unordered_map<Coordinate<int, 4>, int, 5, 8>;

// D = 4
using CoordsToIndexMap_int_5_int_5_0 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 0>;

using CoordsToIndexMap_int_5_int_5_1 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 1>;

using CoordsToIndexMap_int_5_int_5_2 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 2>;

using CoordsToIndexMap_int_5_int_5_3 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 3>;

using CoordsToIndexMap_int_5_int_5_4 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 4>;

using CoordsToIndexMap_int_5_int_5_5 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 5>;

using CoordsToIndexMap_int_5_int_5_6 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 6>;

using CoordsToIndexMap_int_5_int_5_7 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 7>;

using CoordsToIndexMap_int_5_int_5_8 =
    cuda::unordered_map<Coordinate<int, 5>, int, 5, 8>;

// D = 5
using CoordsToIndexMap_int_6_int_5_0 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 0>;

using CoordsToIndexMap_int_6_int_5_1 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 1>;

using CoordsToIndexMap_int_6_int_5_2 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 2>;

using CoordsToIndexMap_int_6_int_5_3 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 3>;

using CoordsToIndexMap_int_6_int_5_4 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 4>;

using CoordsToIndexMap_int_6_int_5_5 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 5>;

using CoordsToIndexMap_int_6_int_5_6 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 6>;

using CoordsToIndexMap_int_6_int_5_7 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 7>;

using CoordsToIndexMap_int_6_int_5_8 =
    cuda::unordered_map<Coordinate<int, 6>, int, 5, 8>;

// D = 6
using CoordsToIndexMap_int_7_int_5_0 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 0>;

using CoordsToIndexMap_int_7_int_5_1 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 1>;

using CoordsToIndexMap_int_7_int_5_2 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 2>;

using CoordsToIndexMap_int_7_int_5_3 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 3>;

using CoordsToIndexMap_int_7_int_5_4 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 4>;

using CoordsToIndexMap_int_7_int_5_5 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 5>;

using CoordsToIndexMap_int_7_int_5_6 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 6>;

using CoordsToIndexMap_int_7_int_5_7 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 7>;

using CoordsToIndexMap_int_7_int_5_8 =
    cuda::unordered_map<Coordinate<int, 7>, int, 5, 8>;


using CoordsToIndexMapGPU = CoordsToIndexMap_int_4_int_5_5;

template <typename MapType = CoordsToIndexMapGPU> struct GPUCoordsMap {
  shared_ptr<MapType> map;
  using key_type = typename MapType::key_type;
  using value_type = typename MapType::value_type;
  int nrows, ncols;

  // Constructors
  GPUCoordsMap(uint32_t map_size, float duplicate_factor=1.0,
               uint32_t keys_per_bucket=62, const uint32_t device_id=0) {
    /*
    map->reserve(map_size, duplicate_factor,
                keys_per_bucket, device_id);
                */
    map = std::make_shared<MapType>(map_size, duplicate_factor,
                                    keys_per_bucket, device_id);
  }

  // Initializations
  value_type
  initialize_batch(const int* p_coords_,
                   int* p_mapping_,
                   int* p_inverse_mapping_,
                   const int nrows_, const int ncols_,
                   const bool force_remap = false,
                   const bool return_inverse = false);

  void get_coords(int* p_coords, int size);
  void get_index_at_batch(int* p_out, int batch_index, int nrows_);
  void get_index_per_batch(const vector<int*>& p_outs, int nrows_);
  value_type
  //region_insert(const GPUCoordsMap<MapType>& in_coords_map,
  region_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                const Region &region, int size);
  value_type
  region_insert_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                       const vector<int*>& p_ins,
                       const vector<int*>& p_outs,
                       const Region &region,
                       int size);
  void region_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                     const vector<int*>& p_ins,
                     const vector<int*>& p_outs,
                     const Region &region,
                     int size);
  value_type
  batch_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map, int size);
  void batch_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                    int* p_in, int* p_out, int size);
  value_type
  stride_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                const vector<int>& tensor_strides,
                int size);
  value_type
  stride_insert_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                         int* p_in, int* p_out,
                                         const vector<int>& tensor_strides,
                                         int size);
  void stride_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                     int* p_in, int* p_out,
                     const vector<int>& tensor_strides,
                     int size);
  value_type
  union_insert(
    const vector<shared_ptr<GPUCoordsMap<MapType>>>& in_maps,
    const vector<int>& in_coords_map_sizes);
  value_type
  union_insert_search(
    const vector<shared_ptr<GPUCoordsMap<MapType>>>& in_maps,
    const vector<int*>& p_ins, const vector<int*>& p_outs,
    const vector<int>& in_coords_map_sizes);
  void union_search(
    const vector<shared_ptr<GPUCoordsMap<MapType>>>& in_maps,
    const vector<int*>& p_ins, const vector<int*>& p_outs,
    const vector<int>& in_coords_map_sizes);
  value_type
  prune_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                      bool* p_keep, int keep_size,
                                      int size);
  value_type
  prune_insert_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                           int* p_in, int* p_out,
                                           bool* p_keep, int keep_size,
                                           int size);
  void
  prune_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                    int* p_in, int* p_out,
                                    bool* p_keep, int keep_size,
                                    int size);
  size_t size() const {
    ASSERT(map->Size() == nrows, "map->Size() should equal to nrows");
    return nrows;
  }
};

} // end namespace minkowski

#endif // gpu coordsmap
