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
#include "common.hpp"
#include "region.hpp"
#include "utils.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace minkowski {

/*
 * Given tensor_stride_src and tensor_stride_dst, find the respective coord_maps
 * and return the indices of the coord_map_ind in coord_map_dst
 */
template <typename MapType>
vector<vector<at::Tensor>> GPUCoordsManager<MapType>::getKernelMap(
    const vector<int>& tensor_strides, const vector<int>& strides,
    const vector<int>& kernel_sizes,
    const vector<int>& dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    bool is_transpose, bool is_pool) {
  // WARNING: This function will not work properly with custon region types.
  ASSERT(region_type != 2,
         "Currently, it does not support the custom region type.");
  /*
  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  const auto &in_map_iter = in_maps.find(map_key);
  */

//  if (in_map_iter == in_maps.end()) {
  const InOutMapKey map_key = getInOutMaps(tensor_strides, strides, kernel_sizes, dilations, region_type,
                 offsets, py_in_coords_key, py_out_coords_key, false);
//    ASSERT(in_maps.find(map_key) != in_maps.end(), "Kernel map not found.");
//  }

  return {in_maps[map_key], out_maps[map_key]};
}

template <typename MapType>
vector<at::Tensor>
GPUCoordsManager<MapType>::getCoordsMap(py::object py_in_coords_key,
                                     py::object py_out_coords_key) const {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  const uint64_t out_coords_key = p_out_coords_key->getKey();

  const auto in_map_iter = coords_maps.find(in_coords_key);
  const auto out_map_iter = coords_maps.find(out_coords_key);

  ASSERT(in_map_iter != coords_maps.end(), "Input coords not found at",
         to_string(in_coords_key));
  ASSERT(out_map_iter != coords_maps.end(), "Output coords not found at",
         to_string(out_coords_key));

  const auto &out_tensor_strides = p_out_coords_key->getTensorStride();

  const auto nrows = in_map_iter->second->nrows;

  at::Tensor in =
      torch::empty({static_cast<int>(nrows + 1)},
                   torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor out =
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32));

  int* p_in = in.data<int>();
  int* p_out = out.data<int>();

  out_map_iter->second->stride_search(in_map_iter->second,
                                     p_in, p_out,
                                     out_tensor_strides,
                                     nrows);
  int size = *(p_in + nrows);
  in.resize_({size});
  out.resize_({size});
  return {in, out};
}

template <typename MapType>
uint64_t
GPUCoordsManager<MapType>::getCoordsKey(const vector<int> &tensor_strides) const {
  auto tensor_stride_hash = hash_vec<vector<int>>(tensor_strides);
  ASSERT(coords_maps.find(tensor_stride_hash) != coords_maps.end(),
         "The coord map doesn't exist for the given tensor strides ",
         "tensor_stride: ", ArrToString(tensor_strides));
  return tensor_stride_hash;
}

template <typename MapType>
bool GPUCoordsManager<MapType>::existsCoordsKey(const uint64_t coords_key) const {
  return coords_maps.find(coords_key) != coords_maps.end();
}

template <typename MapType>
bool GPUCoordsManager<MapType>::existsCoordsKey(py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  return existsCoordsKey(p_coords_key->getKey());
}

template <typename MapType>
uint64_t GPUCoordsManager<MapType>::getRandomCoordsKey() {
  uint64_t coords_key = random();
  while (coords_maps.find(coords_key) != coords_maps.end())
    coords_key = random();
  return coords_key;
}

template <typename MapType>
int GPUCoordsManager<MapType>::getCoordsSize(const uint64_t coords_key) const {
  const auto &coords_map_iter = coords_maps.find(coords_key);
  ASSERT(coords_map_iter != coords_maps.end(),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");
  return coords_map_iter->second->size();
}

template <typename MapType>
int GPUCoordsManager<MapType>::getCoordsSize(py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  return getCoordsSize(p_coords_key->getKey());
}

template <typename MapType>
void GPUCoordsManager<MapType>::getCoords(at::Tensor coords,
                                       py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  const uint64_t coords_key = p_coords_key->getKey();

  // initialize
  const auto &coords_map_iter = coords_maps.find(coords_key);
  ASSERT(coords_map_iter != coords_maps.end(),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");
  //const GPUCoordsMap<MapType> &coordmap = coords_map_iter->second;
//  const auto& coordmap = coords_map_iter->second;
  /*
  int nrows = coordmap->nrows;
  int ncols = coordmap->ncols;
  */
  int nrows = coords_map_iter->second->nrows;
  int ncols = coords_map_iter->second->ncols;
  coords.resize_({nrows, ncols});
  int *p_coords = coords.data<int>();

  //coordmap->get_coords(p_coords, nrows);
  coords_map_iter->second->get_coords(p_coords, nrows);
}

template <typename MapType>
void GPUCoordsManager<MapType>::setOriginCoordsKey(py::object py_coords_key) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  const int D = p_coords_key->getDimension();
  ASSERT(D > 0, "Invalid dimension: ", D);
  if (!p_coords_key->key_set) {
    p_coords_key->setKey(createOriginCoords(D));
    const vector<int> zero_vec(D, 0);
    p_coords_key->setTensorStride(zero_vec);
  } else {
    auto coords_key = p_coords_key->getKey();
    auto origin_key = createOriginCoords(D);
    ASSERT(coords_key == origin_key, "Invalid key: ", to_string(coords_key),
           " != Origin key: ", to_string(origin_key));
  }
}

/*******************************
 * Initialization
 *******************************/

/*
 * coords: coordinates in IntTensor
 * mapping: output mapping in IntTensor
 * tensor_strides: current tensor strides this coords will be initializeds
 * force_creation: even when there's a duplicate coords with the same tensor
 *                 strides.
 * force_remap: if there's duplicate coords, remap
 * allow_duplicate_coords: create map when there are duplicates in the
 * coordinates
 */
template <typename MapType>
uint64_t GPUCoordsManager<MapType>::initializeCoords(
    at::Tensor coords, at::Tensor mapping, at::Tensor inverse_mapping,
    const vector<int> &tensor_strides, const bool force_creation,
    const bool force_remap, const bool allow_duplicate_coords,
    const bool return_inverse) {
  device = coords.device();
  const int nrows = coords.size(0);
  const int ncols = coords.size(1);
  const int D = ncols - 1;

  // Basic assertions
  ASSERT(force_creation == true, "force_creation must be true");
  ASSERT(D == tensor_strides.size(), "The coordinate dimension (ncols - 1) ",
         to_string(D),
         " must match the size of tensor stride: ", ArrToString(tensor_strides),
         ".");

  uint64_t key = hash_vec(tensor_strides);

  if (coords_maps.find(key) != coords_maps.end()) {
    // If force creation, set a random key that doesn't exist
    if (force_creation) {
      key = getRandomCoordsKey();
    } else {
      ASSERT(false, "The coord map already exists for the given tensor stride ",
             "tensor_stride: ", ArrToString(tensor_strides),
             "For more information, please refer to the SparseTensor creation "
             "documentation available at:"
             "https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html");
    }
  }

  // Create the concurrent coords map
  mapping.resize_(static_cast<int>(nrows)).to(device);
  inverse_mapping.resize_(static_cast<int>(nrows)).to(device);
  int* p_coords = coords.data<int>();
  int* p_mapping = mapping.data<int>();
  int* p_inverse_mapping = inverse_mapping.data<int>();
  float duplicate_factor = 0.1;
  coords_maps[key] = std::make_shared<GPUCoordsMap<MapType>>(nrows, duplicate_factor);

  ASSERT(force_remap == true,
         "Please use cpu version when force_remap == false");

  auto coords_map_size = coords_maps[key]->initialize_batch(
      p_coords, p_mapping, p_inverse_mapping,
      nrows, ncols, force_remap, return_inverse);

  min_nrows = coords_map_size;
  min_coords_key = key;

  if (!allow_duplicate_coords && !force_remap) {
    ASSERT(nrows == coords_map_size, "Duplicate coordinates found. ",
           "Number of input coords:", nrows,
           " != Number of unique coords:", coords_map_size,
           "If the duplication was intentional, set force_remap to true."
           "For more information, please refer to the SparseTensor creation "
           "documentation available at: "
           "https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html");
  }

  // When remapping, return the mapping to pytorch.
  if (force_remap || return_inverse) {
//    ASSERT(mapping.dtype() == torch::kInt64,
//           "Mapping must be a torch::LongTensor");
    mapping.resize_({coords_map_size});
  }

  if (return_inverse) {
//    ASSERT(inverse_mapping.dtype() == torch::kInt64,
//           "Inverse Mapping must be a torch::LongTensor");
    ASSERT(inverse_mapping.size(0) == nrows,
           "inverse_mapping's size must equal to nrows");
  }

  return key;
}

template <typename MapType>
uint64_t GPUCoordsManager<MapType>::initializeCoords(
    at::Tensor coords, at::Tensor mapping, at::Tensor inverse_mapping,
    py::object py_coords_key, const bool force_creation, const bool force_remap,
    const bool allow_duplicate_coords, const bool return_inverse) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();

  const uint64_t in_coords_key = initializeCoords(
      coords, mapping, inverse_mapping, p_coords_key->getTensorStride(),
      force_creation, force_remap, allow_duplicate_coords, return_inverse);

  // Tensor strides initialized on the python side.
  p_coords_key->setKey(in_coords_key);

  return in_coords_key;
}

/*********************************/
template <typename MapType>
uint64_t GPUCoordsManager<MapType>::createStridedCoords(
    uint64_t coords_key, const vector<int> &tensor_strides,
    const vector<int> &strides, bool force_creation) {
  // Basic assertions
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, false);

  const int D = coords_maps[coords_key]->ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "GPUCoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  uint64_t out_coords_key = 0;
  const bool is_identity =
      std::all_of(strides.begin(), strides.end(), [](int s) { return s == 1; });

  if (is_identity) {
    out_coords_key = coords_key;
  } else {

    // tensor_strides.size() == strides.size() on computeOutTensorStride
    out_coords_key = hash_vec(out_tensor_strides);

    // If force creationg, get a random key.
    // ElseIf the coordinates already exists, return the key.
    if (force_creation) {
      if (existsCoordsKey(out_coords_key))
        out_coords_key = getRandomCoordsKey();
    } else if (existsCoordsKey(out_coords_key)) {
      return out_coords_key;
    }

    // Create a strided coords map
    int duplicate_factor = 1;
    for (auto stride : strides) duplicate_factor *= stride;
    duplicate_factor = 1.0 / duplicate_factor;
    const auto nrows = coords_maps[coords_key]->nrows;
    coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(nrows, duplicate_factor);
    auto out_nrows = coords_maps[out_coords_key]->stride_insert(coords_maps[coords_key],
                                                               out_tensor_strides,
                                                               nrows);
    if (out_nrows < min_nrows) {
      min_nrows = out_nrows;
      min_coords_key = out_coords_key;
    }
  }

  return out_coords_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getStridedInOutMaps(
    py::object py_in_coords_key, py::object py_out_coords_key,
    const vector<int>& tensor_strides, const vector<int>& strides,
    const vector<int>& kernel_sizes, const vector<int>& dilations, int region_type,
    bool is_transpose, bool is_pool,
    bool force_creation) {

  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t out_coords_key = 0;

  /*
  if (!p_out_coords_key->tensor_stride_set) {
    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->up_stride(strides);
  }
  */

  // Basic assertions
  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  const int D = coords_maps[in_coords_key]->ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "GPUCoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, is_transpose);

  const bool is_identity =
      std::all_of(strides.begin(), strides.end(), [](int s) { return s == 1; });

  if (is_identity) {
    ASSERT(!p_out_coords_key->isKeySet() ||
            p_out_coords_key->getKey() == p_in_coords_key->getKey(),
           "Be aware of coords_key overwrite leakage");
    out_coords_key = in_coords_key;
    p_out_coords_key->setKey(out_coords_key);
    if (!p_out_coords_key->tensor_stride_set) {
      p_out_coords_key->setTensorStride(tensor_strides);
      p_out_coords_key->up_stride(strides);
    }
  } else if (force_creation) {
    return createStridedInOutMaps(
        py_in_coords_key, py_out_coords_key,
        tensor_strides, strides,
        kernel_sizes, dilations, region_type,
        is_transpose, is_pool,
        true);
  } else if (p_out_coords_key->isKeySet()) {
    out_coords_key = p_out_coords_key->getKey();
  } else {
    out_coords_key = hash_vec(out_tensor_strides);
    if (!existsCoordsKey(out_coords_key)) {
      return createStridedInOutMaps(
          py_in_coords_key, py_out_coords_key,
          tensor_strides, strides,
          kernel_sizes, dilations, region_type,
          is_transpose, is_pool,
          false);
    }
  }

  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  if (in_maps.find(map_key) != in_maps.end()) return map_key;

  const auto nrows = coords_maps[in_coords_key]->nrows;

  vector<at::Tensor> th_ins(1,
      torch::empty({static_cast<int>(nrows + 1)},
                   torch::TensorOptions().dtype(torch::kInt32)));
  vector<at::Tensor> th_outs(1,
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32)));

  int* p_in = th_ins[0].data<int>();
  int* p_out = th_outs[0].data<int>();

  coords_maps[out_coords_key]->stride_search(coords_maps[in_coords_key],
                                            p_in, p_out,
                                            out_tensor_strides,
                                            nrows);
  int size = *(p_in + nrows);
  th_ins[0].resize_({size});
  th_outs[0].resize_({size});
  in_maps[map_key] = move(th_ins);
  out_maps[map_key] = move(th_outs);
  return map_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::createStridedInOutMaps(
    py::object py_in_coords_key, py::object py_out_coords_key,
    const vector<int> &tensor_strides,
    const vector<int> &strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    bool is_transpose, bool is_pool,
    bool force_creation) {

  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t out_coords_key = 0;

  // Basic assertions
  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  const int D = coords_maps[in_coords_key]->ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "GPUCoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  const bool is_identity =
      std::all_of(strides.begin(), strides.end(), [](int s) { return s == 1; });

  ASSERT(is_identity == false,
         "Please check is_identity in getStridedInOutMaps");

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, is_transpose);

  out_coords_key = hash_vec(out_tensor_strides);
  if (force_creation) {
    if (existsCoordsKey(out_coords_key))
      out_coords_key = getRandomCoordsKey();
  } else {
    ASSERT(!existsCoordsKey(out_coords_key),
           "createX will always come from getX, getX has handled this condition");
  }

  p_out_coords_key->setKey(out_coords_key);

  if (!p_out_coords_key->tensor_stride_set) {
    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->up_stride(strides);
  }

  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

//  if (in_maps.find(map_key) != in_maps.end()) return;
  ASSERT(in_maps.find(map_key) == in_maps.end(),
         "out_coords_key is new, ins/outs maps have to be generated.");

  const auto nrows = coords_maps[in_coords_key]->nrows;

  vector<at::Tensor> th_ins(1,
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32)));
  vector<at::Tensor> th_outs(1,
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32)));

  int* p_in = th_ins[0].data<int>();
  int* p_out = th_outs[0].data<int>();

  // Create a strided coords map
  int duplicate_factor = 1;
  for (auto stride : strides) duplicate_factor *= stride;
  duplicate_factor = 1.0 / duplicate_factor;
  coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(nrows, duplicate_factor);
  auto out_nrows = coords_maps[out_coords_key]->stride_insert_search(coords_maps[in_coords_key],
                                                   p_in, p_out,
                                                   out_tensor_strides,
                                                   nrows);
  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  in_maps[map_key] = move(th_ins);
  out_maps[map_key] = move(th_outs);
  return map_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getTransposedStridedRegionInOutMaps(
    py::object py_in_coords_key, py::object py_out_coords_key,
    const vector<int>& tensor_strides,
    const vector<int>& strides, const vector<int>& kernel_sizes, const vector<int>& dilations,
    int region_type,
    bool is_transpose, bool is_pool,
    at::Tensor offsets,
    bool force_creation) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t out_coords_key = 0;

  // Basic assertions
  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  const int D = coords_maps[in_coords_key]->ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "GPUCoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, is_transpose);

  const bool is_identity =
      std::all_of(strides.begin(), strides.end(), [](int s) { return s == 1; });

  if (is_identity) {
    ASSERT(!p_out_coords_key->isKeySet() ||
            p_out_coords_key->getKey() == p_in_coords_key->getKey(),
           "Be aware of coords_key overwrite leakage");
    out_coords_key = in_coords_key;
    p_out_coords_key->setKey(out_coords_key);
    if (!p_out_coords_key->tensor_stride_set) {
      p_out_coords_key->setTensorStride(tensor_strides);
      p_out_coords_key->up_stride(strides);
    }
  } else if (force_creation) {
    return createTransposedStridedRegionInOutMaps(
        py_in_coords_key, py_out_coords_key,
        tensor_strides, strides,
        kernel_sizes, dilations, region_type,
        is_transpose, is_pool,
        offsets,
        true);
  } else if (p_out_coords_key->isKeySet()) {
    out_coords_key = p_out_coords_key->getKey();
  } else {
    out_coords_key = hash_vec(out_tensor_strides);
    if (!existsCoordsKey(out_coords_key)) {
      return createTransposedStridedRegionInOutMaps(
          py_in_coords_key, py_out_coords_key,
          tensor_strides, strides,
          kernel_sizes, dilations, region_type,
          is_transpose, is_pool,
          offsets,
          false);
    }
  }

  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  if (in_maps.find(map_key) != in_maps.end()) return map_key;

  const auto nrows = coords_maps[in_coords_key]->nrows;

  // Create transposed coords map
  Region region = Region(out_tensor_strides, kernel_sizes, dilations,
                         region_type, offsets.data<int>(), offsets.size(0));

//  in_maps[map_key] = vector<at::Tensor>(region.size(),
  vector<at::Tensor> th_ins(region.size(),
      torch::empty({static_cast<int>(nrows + 1)},
                   torch::TensorOptions().dtype(torch::kInt32)));
//  out_maps[map_key] = vector<at::Tensor>(region.size(),
  vector<at::Tensor> th_outs(region.size(),
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32)));

  vector<int*> p_ins(region.size());
  vector<int*> p_outs(region.size());
  for (size_t c = 0; c != region.size(); ++c) {
    p_ins[c] = th_ins[c].data<int>();
    p_outs[c] = th_outs[c].data<int>();
  }

  coords_maps[out_coords_key]->region_search(coords_maps[in_coords_key],
                                            p_ins, p_outs,
                                            region, nrows);
  for (size_t c = 0; c != region.size(); ++c) {
    int size = *(p_ins[c] + nrows);
    th_ins[c].resize_({size});
    th_outs[c].resize_({size});
  }
  in_maps[map_key] = move(th_ins);
  out_maps[map_key] = move(th_outs);
  return map_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::createTransposedStridedRegionInOutMaps(
    py::object py_in_coords_key, py::object py_out_coords_key,
    const vector<int>& tensor_strides,
    const vector<int>& strides, const vector<int>& kernel_sizes, const vector<int>& dilations,
    int region_type,
    bool is_transpose, bool is_pool,
    at::Tensor offsets, bool force_creation) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t out_coords_key = 0;

  // Basic assertions
  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  const int D = coords_maps[in_coords_key]->ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "GPUCoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  const bool is_identity =
      std::all_of(strides.begin(), strides.end(), [](int s) { return s == 1; });

  ASSERT(is_identity == false,
         "Please check is_identity in getStridedInOutMaps");

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, is_transpose);

  // Set the out_coords_key and return if a key already exists.
  out_coords_key = hash_vec(out_tensor_strides);
  if (force_creation) {
    // set a random coords key if force creation is set
    if (existsCoordsKey(out_coords_key))
      out_coords_key = getRandomCoordsKey();
  } else {
    ASSERT(!existsCoordsKey(out_coords_key),
           "createX will always come from getX, getX has handled this condition");
  }

  p_out_coords_key->setKey(out_coords_key);

  if (!p_out_coords_key->tensor_stride_set) {
    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->up_stride(strides);
  }

  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

//  if (in_maps.find(map_key) != in_maps.end()) return;
  ASSERT(in_maps.find(map_key) == in_maps.end(),
         "out_coords_key is new, ins/outs maps have to be generated.");

  const auto nrows = coords_maps[in_coords_key]->nrows;

  // Create transposed coords map
  Region region = Region(out_tensor_strides, kernel_sizes, dilations,
                         region_type, offsets.data<int>(), offsets.size(0));

//  in_maps[map_key] = vector<at::Tensor>(region.size(),
  vector<at::Tensor> th_ins(region.size(),
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32)));
//  out_maps[map_key] = vector<at::Tensor>(region.size(),
  vector<at::Tensor> th_outs(region.size(),
      torch::empty({static_cast<int>(nrows)},
                   torch::TensorOptions().dtype(torch::kInt32)));

  vector<int*> p_ins(region.size());
  vector<int*> p_outs(region.size());
  for (size_t c = 0; c != region.size(); ++c) {
    p_ins[c] = th_ins[c].data<int>();
    p_outs[c] = th_outs[c].data<int>();
  }

  float duplicate_factor = 1.0;
  for (auto stride : strides) duplicate_factor *= stride;
  coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(
                                    nrows,
                                    duplicate_factor);
  auto out_nrows = coords_maps[out_coords_key]->region_insert_search(
                                    coords_maps[in_coords_key],
                                    p_ins, p_outs,
                                    region, nrows);
  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  in_maps[map_key] = move(th_ins);
  out_maps[map_key] = move(th_outs);
  return map_key;
}

template <typename MapType>
uint64_t GPUCoordsManager<MapType>::createTransposedStridedRegionCoords(
    uint64_t coords_key, const vector<int> &tensor_strides,
    const vector<int> &strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, at::Tensor offsets, bool force_creation) {
  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, true /* is_transpose */);

  // Basic assertions
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");

  const int D = coords_maps[coords_key]->ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "GPUCoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  // Set the out_coords_key and return if a key already exists.
  uint64_t out_coords_key = hash_vec(out_tensor_strides);
  if (force_creation) {
    // set a random coords key if force creation is set
    if (existsCoordsKey(out_coords_key))
      out_coords_key = getRandomCoordsKey();
  } else if (existsCoordsKey(out_coords_key)) {
    // Returnn if not force_creation and the key exists
    return out_coords_key;
  }

  // Create transposed coords map
  Region region = Region(out_tensor_strides, kernel_sizes, dilations,
                         region_type, offsets.data<int>(), offsets.size(0));

  const int nrows = coords_maps[coords_key]->nrows;
  float duplicate_factor = 1.0;
  for (auto stride : strides) duplicate_factor *= stride;
  coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(nrows,
                                                   duplicate_factor);
  auto out_nrows = coords_maps[out_coords_key]->region_insert(
                              coords_maps[coords_key], region, nrows);

  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  return out_coords_key;
}

template <typename MapType>
uint64_t GPUCoordsManager<MapType>::createOriginCoords(const int D) {
  const vector<int> zero_tensor_strides(D, 0);
  const uint64_t out_coords_key = hash_vec(zero_tensor_strides);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(1, 1.0);
  // TODO(ljm): implement batch_insert
  batch_size = coords_maps[out_coords_key]->batch_insert(coords_maps[min_coords_key],
                                                        coords_maps[min_coords_key]->nrows);
  if (batch_size < min_nrows) {
    min_nrows = batch_size;
    min_coords_key = out_coords_key;
  }
  return out_coords_key;
}

template <typename MapType>
long int GPUCoordsManager<MapType>::getBatchSize() {
  if (batch_size == -1) createOriginCoords(D);
  return batch_size;
}

template <typename MapType>
const InOutMapKey GPUCoordsManager<MapType>::getMapHashKey(
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose, bool is_pool) const {
  const int D = tensor_strides.size();
  ASSERT(D == tensor_strides.size() and D == strides.size() and
             D == kernel_sizes.size() and D == dilations.size(),
         "Size mismatch. tensor_strides: ", tensor_strides.size(),
         ", strides: ", strides.size(), ", kernel_sizes: ", kernel_sizes.size(),
         ", dilations: ", dilations.size());

  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t out_coords_key = p_out_coords_key->getKey();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  const uint64_t stride_hash = hash_vec(strides);
  const uint64_t kernel_size_hash = hash_vec(kernel_sizes);
  const uint64_t dilation_hash = hash_vec(dilations);
  const InOutMapKey map_key = {
      in_coords_key, out_coords_key,        stride_hash,  kernel_size_hash,
      dilation_hash, (uint64_t)region_type, is_transpose, is_pool};

  return map_key;
}

template <typename MapType>
const InOutMapKey GPUCoordsManager<MapType>::getOriginMapHashKey(
    py::object py_in_coords_key, py::object py_out_coords_key) const {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  ASSERT(
      p_in_coords_key->key_set and p_out_coords_key->key_set,
      "Key is not set. in_coords_key: ", to_string(p_in_coords_key->getKey()),
      ", out_coords_key: ", to_string(p_out_coords_key->getKey()));

  const int D = p_in_coords_key->getDimension();

  const uint64_t out_coords_key = p_out_coords_key->getKey();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  const vector<int> zero_vec(D, 0);
  const uint64_t zero_hash = hash_vec(zero_vec);
  const InOutMapKey map_key = {
      in_coords_key, out_coords_key, zero_hash, zero_hash, zero_hash, 0, false,
      true};
  return map_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getUnionMapHashKey(vector<py::object> py_in_coords_keys,
                                           py::object py_out_coords_key) const {
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  ASSERT(py_in_coords_keys.size() > 1, "Number of input coords must be > 1");
  vector<CoordsKey *> p_in_coords_keys;
  // We use sum of coords key (even with overflow, it will be unique with high
  // prob). We use sum to make the key invariant to the order of the keys.
  uint64_t sum_in_coords_key = 0;
  CoordsKey *p_in_coords_key = py_in_coords_keys[0].cast<CoordsKey *>();
  for (auto &py_in_coords_key : py_in_coords_keys) {
    p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
    const uint64_t in_coords_key = p_in_coords_key->getKey();
    ASSERT(existsCoordsKey(in_coords_key),
           "The coord map doesn't exist for the given coords_key: ",
           to_string(in_coords_key), ".");
    sum_in_coords_key += in_coords_key;
  }

  ASSERT(p_out_coords_key->key_set, "Key is not set. out_coords_key: ",
         to_string(p_out_coords_key->getKey()));

  const uint64_t out_coords_key = p_out_coords_key->getKey();
  const vector<int> zero_vec(p_in_coords_key->getDimension(), 0);
  const uint64_t zero_hash = hash_vec(zero_vec);
  InOutMapKey map_key = {sum_in_coords_key,
                         out_coords_key,
                         zero_hash,
                         zero_hash,
                         zero_hash,
                         0,
                         false,
                         true};
  return map_key;
}
/**
 * Entry function for coords map generation and the associated kernel maps.
 */
template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getInOutMaps(
    const vector<int> &tensor_strides, const vector<int> &strides,
    const vector<int> &kernel_sizes, const vector<int> &dilations,
    int region_type, const at::Tensor &offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose, bool is_pool,
    bool force_creation) {
  //
  // Warning(ljm): In the GPU version, when `is_transpose == True`,
  // the `Filp` ins/outs maps generation in CPU is not used.
  // It is same as the `non-Flip` version except when `is_pool == True`
  // and `kernel_size` is even, there will be a little bit of difference
  // of the ins/outs maps. But it does not affect the math meaning,
  // just like the customized `region` based `Sparse Convolution`
  // compared with the classic version similar as the normal
  // `Convolution` in torch as `spconv` does.
  // By the way, the `non-Flip` GPU implementation could merge the
  // insert and search operation which reduce the region iteration
  // times from two to one which should be a big optimization.
  //
  // Another remark:
  // By the logic of the implementation in CPU version,
  // the following ins/outs flip cache will never hit.
  //  in_maps[map_key] = out_maps[tmp_map_key];
  //  out_maps[map_key] = in_maps[tmp_map_key];
  // So we do not check it in GPU version.
  // By the way, if it hits, the `non-Flip` implementation in GPU version
  // will catch it automatically. Although it will never hit.
  //
  const int D = tensor_strides.size();
  ASSERT(D == tensor_strides.size() and D == strides.size() and
             D == kernel_sizes.size() and D == dilations.size(),
         "Size mismatch. tensor_strides: ", tensor_strides.size(),
         ", strides: ", strides.size(), ", kernel_sizes: ", kernel_sizes.size(),
         ", dilations: ", dilations.size());
  ASSERT(std::all_of(tensor_strides.begin(), tensor_strides.end(),
                     [](int k) { return k > 0; }),
         "Invalid tensor_strides: ", ArrToString(tensor_strides),
         " Tensor strides must be positive integers.");

    if (!is_transpose) {
      // TODO(ljm): track the update in cpu version
      // TODO: even numbered kernel size to use region_type 0
      if (is_pool && (strides == kernel_sizes)) {
        return getStridedInOutMaps(
            py_in_coords_key, py_out_coords_key,
            tensor_strides, strides,
            kernel_sizes, dilations, region_type,
            is_transpose, is_pool,
            force_creation);
      } else {
        CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
        CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
        // Will return the in_coords_key if strides == 1.
        auto out_coords_key = createStridedCoords(
            p_in_coords_key->getKey(), tensor_strides, strides, force_creation);

        p_out_coords_key->setKey(out_coords_key);
        if (!p_out_coords_key->tensor_stride_set) {
          p_out_coords_key->setTensorStride(tensor_strides);
          p_out_coords_key->up_stride(strides);
        }

        // use Tranposed to generate Non-Tranpose
        // Flip is needed.
        // But, Filp equals to Non-Flip when kernel is symmetric.
        // And, it has little differece when kernel is non-symmetric.
        return getTransposedStridedRegionInOutMaps(
            py_in_coords_key, py_out_coords_key,
            tensor_strides, strides, kernel_sizes,
            dilations, region_type,
            is_transpose, is_pool,
            offsets, false);
      }
    } else {
      const bool is_identity =
          std::all_of(strides.begin(), strides.end(), [](int s) { return s == 1; });
      ASSERT(is_identity == false,
          "It is meaningless of identity in transpose conv");

      if (is_pool && strides == kernel_sizes && region_type == 0) {
        CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
        CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
        auto out_coords_key = createTransposedStridedRegionCoords(
            p_in_coords_key->getKey(), tensor_strides, strides, kernel_sizes,
            dilations, region_type, offsets, force_creation);

        p_out_coords_key->setKey(out_coords_key);
        if (!p_out_coords_key->tensor_stride_set) {
          p_out_coords_key->setTensorStride(tensor_strides);
          p_out_coords_key->up_stride(strides);
        }

        return getStridedInOutMaps(
            py_in_coords_key, py_out_coords_key,
            tensor_strides, strides,
            kernel_sizes, dilations, region_type,
            is_transpose, is_pool,
            false);
      } else {
        return getTransposedStridedRegionInOutMaps(
            py_in_coords_key, py_out_coords_key,
            tensor_strides, strides, kernel_sizes,
            dilations, region_type,
            is_transpose, is_pool,
            offsets, force_creation);
      }
    }
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getOriginInOutMaps(py::object py_in_coords_key,
                                           py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  const int D = p_in_coords_key->getDimension();
  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    p_out_coords_key->setKey(createOriginCoords(D));
    const vector<int> zero_vec(D, 0);
    p_out_coords_key->setTensorStride(zero_vec);
  }

  const uint64_t in_coords_key = p_in_coords_key->getKey();
  const uint64_t out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    ASSERT(coords_maps[out_coords_key]->size() == batch_size,
           "Coords size mismatch. GPUCoordsMap size: ",
           coords_maps[out_coords_key]->size(),
           ", batch size: ", batch_size);
    const auto nrows = coords_maps[in_coords_key]->nrows;
    vector<at::Tensor> th_ins(1, torch::empty(
              {static_cast<int>(nrows)},
              torch::TensorOptions().dtype(torch::kInt32)));
    vector<at::Tensor> th_outs(1, torch::empty(
              {static_cast<int>(nrows)},
              torch::TensorOptions().dtype(torch::kInt32)));
    int* p_in = th_ins[0].data<int>();
    int* p_out = th_outs[0].data<int>();
    coords_maps[out_coords_key]->batch_search(
                                coords_maps[in_coords_key],
                                p_in, p_out, nrows);
    in_maps[map_key] = move(th_ins);
    out_maps[map_key] = move(th_outs);
  }
  return map_key;
}

template <typename MapType>
pair<vector<at::Tensor>, vector<at::Tensor>>
GPUCoordsManager<MapType>::getUnionMap(vector<py::object> py_in_coords_keys,
                                    py::object py_out_coords_key) {

  // all exception handling will be done inside the following
  const auto map_key = getUnionInOutMaps(py_in_coords_keys, py_out_coords_key);

  return {in_maps[map_key], out_maps[map_key]};
}

// WARNING(ljm): this is not a in-use function
template <typename MapType>
uint64_t
GPUCoordsManager<MapType>::createUnionCoords(vector<py::object> py_in_coords_keys,
                                          py::object py_out_coords_key) {

  //vector<reference_wrapper<GPUCoordsMap<MapType>>> in_coords_maps(py_in_coords_keys.size());
  vector<std::shared_ptr<GPUCoordsMap<MapType>>> in_coords_maps(py_in_coords_keys.size());
  vector<int> in_coords_map_sizes(py_in_coords_keys.size());
  CoordsKey *p_in_coords_key = py_in_coords_keys[0].cast<CoordsKey *>();
  auto tensor_strides = p_in_coords_key->getTensorStride();
  //GPUCoordsMap<MapType>& curr_map = coords_maps[p_in_coords_key->getKey()];
  in_coords_maps[0] = coords_maps[p_in_coords_key->getKey()];
  in_coords_map_sizes[0] = in_coords_maps[0]->nrows;
  int total_in_keys = in_coords_map_sizes[0];
  for (size_t i = 1; i != py_in_coords_keys.size(); ++i) {
    // Set the tensor strides to the smallest elements.
    p_in_coords_key = py_in_coords_keys[i].cast<CoordsKey *>();
    transform(tensor_strides.begin(),                            /* In1 begin */
              tensor_strides.end(),                              /* In1 end */
              p_in_coords_key->getTensorStride().begin(),        /* In2 begin */
              tensor_strides.begin(),                            /* out begin */
              [](int a, int b) -> int { return std::min(a, b); } /* binary op */
    );
    in_coords_maps[i] = coords_maps[p_in_coords_key->getKey()];
    in_coords_map_sizes[i] = in_coords_maps[i]->nrows;
    total_in_keys += in_coords_map_sizes[i];

    const uint64_t in_coords_key = p_in_coords_key->getKey();
    ASSERT(existsCoordsKey(in_coords_key),
           "The coord map doesn't exist for the given coords_key: ",
           to_string(in_coords_key), ".");
  }
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  // set a random coords key
  const uint64_t out_coords_key = getRandomCoordsKey();

  // Set the pycoordskey using the last coords_key
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  p_out_coords_key->setKey(out_coords_key);
  p_out_coords_key->setTensorStride(tensor_strides);

  coords_maps[out_coords_key] =
      std::make_shared<GPUCoordsMap<MapType>>(total_in_keys, 1.0 / in_coords_map_sizes.size());

  auto out_nrows = coords_maps[out_coords_key]->union_insert(in_coords_maps,
                                           in_coords_map_sizes);

  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  return out_coords_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::createUnionInOutMaps(const vector<py::object>& py_in_coords_keys,
                                    py::object py_out_coords_key) {

  //vector<reference_wrapper<GPUCoordsMap<MapType>>> in_coords_maps(py_in_coords_keys.size());
  vector<std::shared_ptr<GPUCoordsMap<MapType>>> in_coords_maps(py_in_coords_keys.size());
  vector<int> in_coords_map_sizes(py_in_coords_keys.size());
  vector<at::Tensor> th_ins(py_in_coords_keys.size());
  vector<at::Tensor> th_outs(py_in_coords_keys.size());
  vector<int*> p_ins(py_in_coords_keys.size());
  vector<int*> p_outs(py_in_coords_keys.size());
  CoordsKey *p_in_coords_key = py_in_coords_keys[0].cast<CoordsKey *>();
  auto tensor_strides = p_in_coords_key->getTensorStride();
  in_coords_maps[0] = coords_maps[p_in_coords_key->getKey()];
  in_coords_map_sizes[0] = in_coords_maps[0]->nrows;
  int total_in_keys = in_coords_map_sizes[0];
  th_ins[0] = torch::empty(
        {static_cast<int>(in_coords_map_sizes[0])}, torch::TensorOptions().dtype(torch::kInt32));
  th_outs[0] = torch::empty(
        {static_cast<int>(in_coords_map_sizes[0])}, torch::TensorOptions().dtype(torch::kInt32));
  p_ins[0] = th_ins[0].data<int>();
  p_outs[0] = th_outs[0].data<int>();
  for (size_t i = 1; i != py_in_coords_keys.size(); ++i) {
    // Set the tensor strides to the smallest elements.
    p_in_coords_key = py_in_coords_keys[i].cast<CoordsKey *>();
    transform(tensor_strides.begin(),                            /* In1 begin */
              tensor_strides.end(),                              /* In1 end */
              p_in_coords_key->getTensorStride().begin(),        /* In2 begin */
              tensor_strides.begin(),                            /* out begin */
              [](int a, int b) -> int { return std::min(a, b); } /* binary op */
    );
    in_coords_maps[i] = coords_maps[p_in_coords_key->getKey()];
    in_coords_map_sizes[i] = in_coords_maps[i]->nrows;
    total_in_keys += in_coords_map_sizes[i];
    th_ins[i] = torch::empty(
          {static_cast<int>(in_coords_map_sizes[i])}, torch::TensorOptions().dtype(torch::kInt32));
    th_outs[i] = torch::empty(
          {static_cast<int>(in_coords_map_sizes[i])}, torch::TensorOptions().dtype(torch::kInt32));
    p_ins[i] = th_ins[i].data<int>();
    p_outs[i] = th_outs[i].data<int>();

    const uint64_t in_coords_key = p_in_coords_key->getKey();
    ASSERT(existsCoordsKey(in_coords_key),
           "The coord map doesn't exist for the given coords_key: ",
           to_string(in_coords_key), ".");
  }
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  // set a random coords key
  const uint64_t out_coords_key = getRandomCoordsKey();

  // Set the pycoordskey using the last coords_key
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  p_out_coords_key->setKey(out_coords_key);
  p_out_coords_key->setTensorStride(tensor_strides);

  coords_maps[out_coords_key] =
      std::make_shared<GPUCoordsMap<MapType>>(total_in_keys, 1.0 / in_coords_map_sizes.size());

  auto out_nrows = coords_maps[out_coords_key]->union_insert_search(in_coords_maps,
                                                  p_ins, p_outs,
                                                  in_coords_map_sizes);

  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  // Map key for origin hash map
  const InOutMapKey map_key =
      getUnionMapHashKey(py_in_coords_keys, py_out_coords_key);

  in_maps[map_key] = move(th_ins);
  out_maps[map_key] = move(th_outs);

  return map_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getUnionInOutMaps(vector<py::object> py_in_coords_keys,
                                          py::object py_out_coords_key) {
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set)
    return createUnionInOutMaps(py_in_coords_keys, py_out_coords_key);

  const uint64_t out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  const InOutMapKey map_key =
    getUnionMapHashKey(py_in_coords_keys, py_out_coords_key);

  if (in_maps.find(map_key) == in_maps.end()) {
    //vector<reference_wrapper<GPUCoordsMap<MapType>>> in_coords_maps(py_in_coords_keys.size());
    vector<std::shared_ptr<GPUCoordsMap<MapType>>> in_coords_maps(py_in_coords_keys.size());
    vector<int> in_coords_map_sizes(py_in_coords_keys.size());
    vector<at::Tensor> th_ins(py_in_coords_keys.size());
    vector<at::Tensor> th_outs(py_in_coords_keys.size());
    vector<int*> p_ins(py_in_coords_keys.size());
    vector<int*> p_outs(py_in_coords_keys.size());
    CoordsKey *p_in_coords_key = py_in_coords_keys[0].cast<CoordsKey *>();
    auto tensor_strides = p_in_coords_key->getTensorStride();
    in_coords_maps[0] = coords_maps[p_in_coords_key->getKey()];
    in_coords_map_sizes[0] = in_coords_maps[0]->nrows;
    int total_in_keys = in_coords_map_sizes[0];
    th_ins[0] = torch::empty(
          {static_cast<int>(in_coords_map_sizes[0])}, torch::TensorOptions().dtype(torch::kInt32));
    th_outs[0] = torch::empty(
          {static_cast<int>(in_coords_map_sizes[0])}, torch::TensorOptions().dtype(torch::kInt32));
    p_ins[0] = th_ins[0].data<int>();
    p_outs[0] = th_outs[0].data<int>();
    for (size_t i = 1; i != py_in_coords_keys.size(); ++i) {
      // Set the tensor strides to the smallest elements.
      p_in_coords_key = py_in_coords_keys[i].cast<CoordsKey *>();
      transform(tensor_strides.begin(),                            /* In1 begin */
                tensor_strides.end(),                              /* In1 end */
                p_in_coords_key->getTensorStride().begin(),        /* In2 begin */
                tensor_strides.begin(),                            /* out begin */
                [](int a, int b) -> int { return std::min(a, b); } /* binary op */
      );
      in_coords_maps[i] = coords_maps[p_in_coords_key->getKey()];
      in_coords_map_sizes[i] = in_coords_maps[i]->nrows;
      total_in_keys += in_coords_map_sizes[i];
      th_ins[i] = torch::empty(
            {static_cast<int>(in_coords_map_sizes[i])}, torch::TensorOptions().dtype(torch::kInt32));
      th_outs[i] = torch::empty(
            {static_cast<int>(in_coords_map_sizes[i])}, torch::TensorOptions().dtype(torch::kInt32));
      p_ins[i] = th_ins[i].data<int>();
      p_outs[i] = th_outs[i].data<int>();

      const uint64_t in_coords_key = p_in_coords_key->getKey();
      ASSERT(existsCoordsKey(in_coords_key),
             "The coord map doesn't exist for the given coords_key: ",
             to_string(in_coords_key), ".");
    }

    coords_maps[out_coords_key]->union_search(in_coords_maps,
                                             p_ins, p_outs,
                                             in_coords_map_sizes);
    in_maps[map_key] = move(th_ins);
    out_maps[map_key] = move(th_outs);
  }

  return map_key;
}

template <typename MapType>
uint64_t
GPUCoordsManager<MapType>::createPruningCoords(at::Tensor use_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  ASSERT(!p_out_coords_key->isKeySet(),
         "p_out_coords_key should be unsetted");

  const uint64_t in_coords_key = p_in_coords_key->getKey();

  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  // set a random coords key
  const uint64_t out_coords_key = getRandomCoordsKey();

  // Set the pycoordskey
  p_out_coords_key->setKey(out_coords_key);
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  if (!p_out_coords_key->tensor_stride_set)
    p_out_coords_key->setTensorStride(p_in_coords_key->getTensorStride());

  coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(use_feat.size(0));
  auto out_nrows = coords_maps[out_coords_key]->prune_insert(coords_maps[in_coords_key],
                                           use_feat.data<bool>(),
                                           use_feat.size(0),
                                           coords_maps[in_coords_key]->nrows);

  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  return out_coords_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::createPruningInOutMaps(at::Tensor use_feat,
                                            py::object py_in_coords_key,
                                            py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  ASSERT(!p_out_coords_key->isKeySet(),
         "p_out_coords_key should be unsetted");

  const uint64_t in_coords_key = p_in_coords_key->getKey();
  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  const uint64_t out_coords_key = getRandomCoordsKey();
  p_out_coords_key->setKey(out_coords_key);
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  if (!p_out_coords_key->tensor_stride_set)
    p_out_coords_key->setTensorStride(p_in_coords_key->getTensorStride());

  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  if (in_maps.find(map_key) != in_maps.end()) return map_key;

  vector<at::Tensor> th_ins(1, torch::empty(
            {static_cast<int>(use_feat.size(0))},
            torch::TensorOptions().dtype(torch::kInt32)));
  vector<at::Tensor> th_outs(1, torch::empty(
            {static_cast<int>(use_feat.size(0))},
            torch::TensorOptions().dtype(torch::kInt32)));

  int* p_in = th_ins[0].data<int>();
  int* p_out = th_outs[0].data<int>();

  coords_maps[out_coords_key] = std::make_shared<GPUCoordsMap<MapType>>(use_feat.size(0));
  auto out_nrows = coords_maps[out_coords_key]->prune_insert_search(coords_maps[in_coords_key],
                                                  p_in, p_out,
                                                  use_feat.data<bool>(),
                                                  use_feat.size(0),
                                                  coords_maps[in_coords_key]->nrows);

  if (out_nrows < min_nrows) {
    min_nrows = out_nrows;
    min_coords_key = out_coords_key;
  }
  in_maps[map_key] = move(th_ins);
  out_maps[map_key] = move(th_outs);
  return map_key;
}

template <typename MapType>
const InOutMapKey
GPUCoordsManager<MapType>::getPruningInOutMaps(at::Tensor use_feat,
                                            py::object py_in_coords_key,
                                            py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    // The following function setup py_out_coords_key
    return createPruningInOutMaps(use_feat, py_in_coords_key, py_out_coords_key);
  }

  const uint64_t in_coords_key = p_in_coords_key->getKey();
  const uint64_t out_coords_key = p_out_coords_key->getKey();

  // Use the map key for origin hash map (stride, dilation, kernel are all
  // NULL)
  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    vector<at::Tensor> th_ins(1, torch::empty(
              {static_cast<int>(use_feat.size(0) + 1)},
              torch::TensorOptions().dtype(torch::kInt32)));
    vector<at::Tensor> th_outs(1, torch::empty(
              {static_cast<int>(use_feat.size(0))},
              torch::TensorOptions().dtype(torch::kInt32)));

    int* p_in = th_ins[0].data<int>();
    int* p_out = th_outs[0].data<int>();

//    coords_maps[out_coords_key] = GPUCoordsMap<MapType>(use_feat.size(0));
    coords_maps[out_coords_key]->prune_search(coords_maps[in_coords_key],
                                             p_in, p_out,
                                             use_feat.data<bool>(),
                                             use_feat.size(0),
                                             coords_maps[in_coords_key]->nrows);
    int size = *(p_in + use_feat.size(0));
    th_ins[0].resize_(size);
    th_outs[0].resize_(size);
    in_maps[map_key] = move(th_ins);
    out_maps[map_key] = move(th_outs);
  }

  return map_key;
}


template <typename MapType> string GPUCoordsManager<MapType>::toString() const {
  Formatter out;
  out << "< GPUCoordsManager\n\tNumber of Coordinate Maps: "
      << to_string(coords_maps.size());
  for (const auto &kv : coords_maps) {
    out << " \n\t\tCoordinate Map Key: " << to_string(kv.first)
        << ", Size: " << to_string((kv.second)->size());
  }
  out << "\n\tNumber of Kernel Maps: " << to_string(in_maps.size());
  for (const auto &kv : in_maps) {
    size_t size = 0;
    for (const auto &map : kv.second)
      size += map.size(0);
    out << " \n\t\tKernel In-Out Map Key: "
        << to_string(hash_vec<InOutMapKey>(kv.first))
        << ", Size: " << to_string(size);
  }
  out << " >\n";
  return out;
}

// TODO(ljm): implement GPUCoordsMap<MapType>::print
/*
template <typename MapType>
void GPUCoordsManager<MapType>::printDiagnostics(py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  const auto &map_iter = coords_maps.find(p_coords_key->getKey());
  ASSERT(map_iter != coords_maps.end(), "Coords map does not exist.");
  map_iter->second.print();
}
*/

/*
 * Return row indices for each batch index
 */
template <typename MapType>
at::Tensor
GPUCoordsManager<MapType>::getRowIndicesAtBatchIndex(py::object py_in_coords_key,
                                                  py::object py_out_coords_key,
                                                  const int batch_index) {
  // py_out_coords_key will be set after the above call.
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  const auto in_coords_key = p_in_coords_key->getKey();
  const auto in_map_iter = coords_maps.find(in_coords_key);
  ASSERT(in_map_iter != coords_maps.end(),
         "The in_coords_key, ", to_string(in_coords_key), ", does not exist.");

  const auto& coordsmap = in_map_iter->second;
  const auto nrows = coordsmap->nrows;
  //const auto batch_num = coordsmap->get_batch_num();
  const auto batch_num = getBatchSize();
  ASSERT(batch_index < batch_num, "batch_index: ", to_string(batch_index),
                       " must smaller than batch_num: ", to_string(batch_num));

  at::Tensor out_ind = torch::zeros(
        {static_cast<int>(nrows + 1)}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  int* p_out_ind = out_ind.data<int>();
  //coordsmap.GetIndexAtBatch(p_out_ind, batch_index);
  coordsmap->get_index_at_batch(p_out_ind, batch_index, nrows);
  int size = *(p_out_ind + nrows);
  out_ind.resize_({size});
  //out_ind.resize_(c10::IntArrayRef(reinterpret_cast<int64_t*>(p_out_ind + nrows), 1));

  return out_ind;
}

/*
 * Return row indices per batch
 */
template <typename MapType>
vector<at::Tensor>
GPUCoordsManager<MapType>::getRowIndicesPerBatch(py::object py_in_coords_key,
                                              py::object py_out_coords_key) {
  // py_out_coords_key will be set after the above call.
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  const auto in_coords_key = p_in_coords_key->getKey();
  const auto in_map_iter = coords_maps.find(in_coords_key);
  ASSERT(in_map_iter != coords_maps.end(),
         "The in_coords_key, ", to_string(in_coords_key), ", does not exist.");

  const auto& coordsmap = in_map_iter->second;
//  const auto batch_num = coordsmap->get_batch_num();
  const auto batch_num = getBatchSize();
  const auto nrows = coordsmap->nrows;
  // Return index.
  vector<at::Tensor> out_inds(batch_num, torch::zeros(
        {static_cast<int>(nrows + 1)}, torch::TensorOptions().dtype(torch::kInt32)));
  vector<int*> p_out_inds(batch_num);
  for (size_t b = 0; b != batch_num; ++b) p_out_inds[b] = out_inds[b].data<int>();
  //coordsmap.GetIndexPerBatch(p_out_inds);
  coordsmap->get_index_per_batch(p_out_inds, nrows);
  for (size_t b = 0; b != batch_num; ++b) {
    int size = *(p_out_inds[b] + nrows);
    out_inds[b].resize_({size});
//    out_inds[b].resize_(c10::IntArrayRef(reinterpret_cast<int64_t*>(p_out_inds[b] + nrows), 1));
  }

  return out_inds;
}

template class GPUCoordsManager<CoordsToIndexMapGPU>;

} // end namespace minkowski
