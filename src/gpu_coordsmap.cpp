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
#include <iostream>
#include <numeric>
#include <omp.h>

#include "gpu_coordsmap.hpp"

namespace minkowski {

/*
 * Use this function when batch_size is setted outside
template <typename MapType>
GPUCoordsMap<MapType>::GPUCoordsMap(int ncols_, int batch_size)
    : nrows(batch_size), ncols(ncols_) {
  map->BulkBatchIndiceInsert(ncols_, batch_size);
}
*/

// TODO(ljm): add prune_insert, prune_insert_search, prune_search

/*
template <typename MapType>
GPUCoordsMap<MapType>::GPUCoordsMap(uint32_t map_size, float duplicate_factor,
                     uint32_t keys_per_bucket=62, const uint32_t device_id=0) {
    // TODO(ljm): add this api
    map->reserve(map_size, duplicate_factor,
                keys_per_bucket, device_id);
}
*/

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::initialize_batch(const int* p_coords,
                                     int* p_mapping,
                                     int* p_inverse_mapping,
                                     const int nrows_,
                                     const int ncols_, const bool force_remap,
                                     const bool return_inverse) {
  nrows = nrows_;
  ncols = ncols_;

  map->BulkInsert(p_coords, p_mapping, p_inverse_mapping, nrows, ncols);

  nrows = map->Size();
  return nrows;
}

template <typename MapType>
void GPUCoordsMap<MapType>::get_coords(int* p_coords, int size) {
    map->IterateKeys(p_coords, size);
}

template <typename MapType>
void GPUCoordsMap<MapType>::get_index_at_batch(int* p_out,
                                            int batch_index,
                                            int nrows_) {
    map->IterateSearchAtBatch(p_out, batch_index, nrows_);
}

template <typename MapType>
void GPUCoordsMap<MapType>::get_index_per_batch(
                                const vector<int*>& p_outs,
                                int nrows_) {
    map->IterateSearchPerBatch(p_outs, nrows_);
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
//GPUCoordsMap<MapType>::region_insert(const GPUCoordsMap<MapType>& in_coords_map,
GPUCoordsMap<MapType>::region_insert(const std::shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                  const Region &region, int size) {
  ASSERT(region.tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  vector<at::Tensor> offsets(region.size(), torch::empty(
        {static_cast<int>(ncols)}, torch::TensorOptions().dtype(torch::kInt32)));
  vector<int> origin(ncols, 0);
  Region cregion(region);
  cregion.set_bounds(origin);
  int c = 0;
  for (const auto& point : cregion) {
    CHECK_CUDA(cudaMemcpy(offsets[c].data<int>(), point.data(),
                          sizeof(int) * ncols,
                          cudaMemcpyHostToDevice));
    map->IterateOffsetInsert(in_coords_map->map,
    //map.IterateOffsetInsert(map,
                            offsets[c].data<int>(),
                            in_coords_map->nrows);
    ++c;
  }
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::region_insert_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                         const vector<int*>& p_ins,
                                         const vector<int*>& p_outs,
                                         const Region &region,
                                         int size) {
  ASSERT(region.tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  vector<at::Tensor> offsets(region.size(), torch::empty(
        {static_cast<int>(ncols)}, torch::TensorOptions().dtype(torch::kInt32)));
  vector<int> origin(ncols, 0);
  Region cregion(region);
  cregion.set_bounds(origin);
  int c = 0;
  for (const auto& point : cregion) {
    CHECK_CUDA(cudaMemcpy(offsets[c].data<int>(), point.data(),
                          sizeof(int) * ncols,
                          cudaMemcpyHostToDevice));
    map->IterateOffsetInsertWithInsOuts(in_coords_map->map,
                                       offsets[c].data<int>(),
                                       p_ins[c], p_outs[c],
                                       size);
    ++c;
  }
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
void
GPUCoordsMap<MapType>::region_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                  const vector<int*>& p_ins,
                                  const vector<int*>& p_outs,
                                  const Region &region,
                                  int size) {
  ASSERT(region.tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  vector<at::Tensor> offsets(region.size(), torch::empty(
        {static_cast<int>(ncols)}, torch::TensorOptions().dtype(torch::kInt32)));
  vector<int> origin(ncols, 0);
  Region cregion(region);
  cregion.set_bounds(origin);
  int c = 0;
  for (const auto& point : cregion) {
    CHECK_CUDA(cudaMemcpy(offsets[c].data<int>(), point.data(),
                          sizeof(int) * ncols,
                          cudaMemcpyHostToDevice));
    map->IterateOffsetSearch(in_coords_map->map,
                            offsets[c].data<int>(),
                            p_ins[c], p_outs[c],
                            size);
    ++c;
  }
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::batch_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                 int size) {
  map->IterateBatchInsert(in_coords_map->map, size);
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
void
GPUCoordsMap<MapType>::batch_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                 int* p_in, int* p_out, int size) {
  map->IterateBatchSearch(in_coords_map->map, p_in, p_out, size);
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::prune_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                    bool* p_keep, int keep_size,
                                    int size) {
  map->IteratePruneInsert(in_coords_map->map, p_keep, keep_size, size);
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::prune_insert_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                         int* p_in, int* p_out,
                                         bool* p_keep, int keep_size,
                                         int size) {
  map->IteratePruneInsertWithInOut(in_coords_map->map,
                                   p_in, p_out,
                                   p_keep, keep_size, size);
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
void
GPUCoordsMap<MapType>::prune_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                  int* p_in, int* p_out,
                                  bool* p_keep, int keep_size,
                                  int size) {
  map->IteratePruneSearch(in_coords_map->map,
                          p_in, p_out,
                          p_keep, keep_size, size);
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::stride_insert(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                  const vector<int>& tensor_strides,
                                  int size) {
  map->IterateStrideInsert(in_coords_map->map, tensor_strides, size);
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::stride_insert_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                         int* p_in, int* p_out,
                                         const vector<int>& tensor_strides,
                                         int size) {
  map->IterateStrideInsertWithInOut(in_coords_map->map,
                                   p_in, p_out,
                                   tensor_strides, size);
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
void
GPUCoordsMap<MapType>::stride_search(const shared_ptr<GPUCoordsMap<MapType>>& in_coords_map,
                                  int* p_in, int* p_out,
                                  const vector<int>& tensor_strides,
                                  int size) {
  map->IterateStrideSearch(in_coords_map->map,
                          p_in, p_out,
                          tensor_strides, size);
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::union_insert(
    const vector<shared_ptr<GPUCoordsMap<MapType>>>& in_maps,
    const vector<int>& in_coords_map_sizes) {
  for (size_t i = 0; i != in_maps.size(); ++i) {
      map->IterateInsert(in_maps[i]->map,
                        in_coords_map_sizes[i]);
  }
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
typename GPUCoordsMap<MapType>::value_type
GPUCoordsMap<MapType>::union_insert_search(
    const vector<shared_ptr<GPUCoordsMap<MapType>>>& in_maps,
    const vector<int*>& p_ins, const vector<int*>& p_outs,
    const vector<int>& in_coords_map_sizes) {
  for (size_t i = 0; i != in_maps.size(); ++i) {
      map->IterateInsertWithInsOuts(in_maps[i]->map, p_ins[i], p_outs[i],
                                   in_coords_map_sizes[i]);
  }
  nrows = map->Size();
  return nrows;
}

template <typename MapType>
void GPUCoordsMap<MapType>::union_search(
    const vector<shared_ptr<GPUCoordsMap<MapType>>>& in_maps,
    const vector<int*>& p_ins, const vector<int*>& p_outs,
    const vector<int>& in_coords_map_sizes) {
  for (size_t i = 0; i != in_maps.size(); ++i) {
      map->IterateSearch(in_maps[i]->map, p_ins[i], p_outs[i],
                        in_coords_map_sizes[i]);
  }
}

// TODO(ljm): add a debug helper function here
/*
template <typename MapType> void GPUCoordsMap<MapType>::print() const {
  for (const auto &kv : map) {
    std::cout << ArrToString(kv.first) << ":" << kv.second << "\n";
  }
  std::cout << std::flush;
}
*/

template struct GPUCoordsMap<CoordsToIndexMapGPU>;
//template struct GPUCoordsMap<CoordsToVectorMap>;

} // end namespace minkowski
