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

/*
 * Given tensor_stride_src and tensor_stride_dst, find the respective coord_maps
 * and return the indices of the coord_map_ind in coord_map_dst
 */
void CoordsManager::getKernelMap(at::Tensor kernel_map,
                                 vector<int> tensor_strides,
                                 vector<int> strides, vector<int> kernel_sizes,
                                 vector<int> dilations, int region_type,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 bool is_transpose, bool is_pool) {
  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  ASSERT(in_maps.find(map_key) != in_maps.end(),
         "The kernelmap does not exist.");

  const InOutMaps<int> &in_map = in_maps[map_key];
  const InOutMaps<int> &out_map = out_maps[map_key];

  int all_volume = 0, kernel_volume = in_map.size();
  for (int k = 0; k < kernel_volume; k++)
    all_volume += in_map[k].size();

  kernel_map.resize_({all_volume, 3});
  int *p_kernel_map = kernel_map.data<int>();

  for (int k = 0; k < kernel_volume; k++) {
    int curr_volume = in_map[k].size();
    for (int i = 0; i < curr_volume; i++) {
      p_kernel_map[0] = k;
      p_kernel_map[1] = in_map[k][i];
      p_kernel_map[2] = out_map[k][i];
      p_kernel_map += 3;
    }
  }
}

uint64_t CoordsManager::getCoordsKey(const vector<int> &tensor_strides) {
  auto tensor_stride_hash = hash_vec<vector<int>>(tensor_strides);
  ASSERT(coords_maps.find(tensor_stride_hash) != coords_maps.end(),
         "The coord map doesn't exist for the given tensor strides ",
         "tensor_stride: ", ArrToString(tensor_strides));
  return tensor_stride_hash;
}

bool CoordsManager::existsCoordsKey(uint64_t coords_key) {
  return coords_maps.find(coords_key) != coords_maps.end();
}

bool CoordsManager::existsCoordsKey(py::object py_coords_key) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  return existsCoordsKey(p_coords_key->getKey());
}

uint64_t CoordsManager::getRandomCoordsKey() {
  uint64_t coords_key = random();
  while (coords_maps.find(coords_key) != coords_maps.end())
    coords_key = random();
  return coords_key;
}

int CoordsManager::getCoordsSize(uint64_t coords_key) {
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");
  return coords_maps[coords_key].size();
}

int CoordsManager::getCoordsSize(py::object py_coords_key) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  return getCoordsSize(p_coords_key->getKey());
}

void CoordsManager::getCoords(at::Tensor coords, py::object py_coords_key) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  uint64_t coords_key = p_coords_key->getKey();

  // initialize
  const auto &coordmap = coords_maps[coords_key];
  int nrows = coordmap.nrows;
  int ncols = coordmap.ncols;
  coords.resize_({nrows, ncols});
  int *p_coords = coords.data<int>();

  // auto &curr_coords = coords_maps[coords_key].coords;
  // copy(curr_coords.begin(), curr_coords.end(), p_coords);

  // copy to the out coords
  for (const auto &kv : coordmap) {
    copy_n(kv.first.begin(), ncols, p_coords + kv.second * ncols);
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
 *                 strides, enforce creation
 * force_remap: if there's duplicate coords, remap
 */
uint64_t CoordsManager::initializeCoords(at::Tensor coords, at::Tensor mapping,
                                         const vector<int> &tensor_strides,
                                         bool force_creation, bool force_remap,
                                         bool allow_duplicate) {
  int nrows = coords.size(0);
  int ncols = coords.size(1);
  int D = ncols - 1;

  // Basic assertions
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
             ". To enforce creation, set true for force_creation.");
    }
  }

  // Create the concurrent coords map
  int *p_coords = coords.data<int>();
  CoordsMap coords_map;
  auto map_batch_pair =
      coords_map.initialize(p_coords, nrows, ncols, force_remap);
  // vector<int> coords_vec(nrows * ncols);
  // copy_n(p_coords, nrows * ncols, coords_vec.data());
  // auto map_batch_pair =
  //     coords_map.initialize(move(coords_vec), nrows, ncols, force_remap);

  // initialize the batch indices
  batch_indices = map_batch_pair.second;

  if (!allow_duplicate && !force_remap) {
    ASSERT(nrows == coords_map.size(), "A duplicate coordinates found. ",
           "If the duplication was intentional, set force_remap to true.");
  }

  // When remapping, return the mapping to pytorch.
  if (force_remap) {
    const vector<int> &map = map_batch_pair.first;
    mapping.resize_({(long)map.size()});
    copy(map.begin(), map.end(), mapping.data<int>());
  }

  // Save the returned results
  coords_maps[key] = move(coords_map);

  return key;
}

uint64_t CoordsManager::initializeCoords(at::Tensor coords, at::Tensor mapping,
                                         py::object py_coords_key,
                                         bool force_creation, bool force_remap,
                                         bool allow_duplicates) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();

  ASSERT(coords.size(1) - 1 == p_coords_key->tensor_strides_.size(),
         "The coordinate dimension - 1, ", coords.size(1) - 1,
         ", mismatches the tensor stride size: ",
         p_coords_key->tensor_strides_.size());

  uint64_t in_coords_key =
      initializeCoords(coords, mapping, p_coords_key->tensor_strides_,
                       force_creation, force_remap, allow_duplicates);

  // Tensor strides initialized on the python side.
  p_coords_key->setKey(in_coords_key);

  return in_coords_key;
}

/*********************************/
uint64_t CoordsManager::createStridedCoords(uint64_t coords_key,
                                            const vector<int> &tensor_strides,
                                            const vector<int> &strides,
                                            bool force_creation) {

  // Basic assertions
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, false);

  int D = coords_maps[coords_key].ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "CoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  // tensor_strides.size() == strides.size() on computeOutTensorStride
  uint64_t out_coords_key = hash_vec(out_tensor_strides);

  // If force creationg, get a random key.
  // ElseIf the coordinates already exists, return the key.
  if (force_creation) {
    out_coords_key = getRandomCoordsKey();
  } else if (existsCoordsKey(out_coords_key)) {
    return out_coords_key;
  }

  // Create a strided coords map
  coords_maps[out_coords_key] =
      coords_maps[coords_key].stride(out_tensor_strides);

  return out_coords_key;
}

uint64_t CoordsManager::createTransposedStridedRegionCoords(
    uint64_t coords_key, const vector<int> &tensor_strides,
    const vector<int> &strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, at::Tensor offsets, bool force_creation) {

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, true /* is_transpose */);

  // Basic assertions
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");

  int D = coords_maps[coords_key].ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "CoordsManager dimension: ", to_string(D),
         ", tensor_strides dimension: ", to_string(tensor_strides.size()));

  // Set the out_coords_key and return if a key already exists.
  uint64_t out_coords_key = hash_vec(out_tensor_strides);
  if (force_creation) {
    // set a random coords key if force creation is set
    out_coords_key = getRandomCoordsKey();
  } else if (existsCoordsKey(out_coords_key)) {
    // Returnn if not force_creation and the key exists
    return out_coords_key;
  }

  // Create transposed coords map
  Region region = Region(out_tensor_strides, kernel_sizes, dilations,
                         region_type, offsets.data<int>(), offsets.size(0));

  coords_maps[out_coords_key] = coords_maps[coords_key].stride_region(region);

  return out_coords_key;
}

uint64_t CoordsManager::createPrunedCoords(at::Tensor use_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  uint64_t in_coords_key = p_in_coords_key->getKey();

  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  // set a random coords key
  uint64_t out_coords_key = getRandomCoordsKey();

  // Set the pycoordskey
  p_out_coords_key->setTensorStride(p_in_coords_key->getTensorStride());
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  p_out_coords_key->setKey(out_coords_key);

  coords_maps[out_coords_key] =
      coords_maps[in_coords_key].prune(use_feat.data<bool>(), use_feat.size(0));

  return out_coords_key;
}

uint64_t CoordsManager::createOriginCoords(int D) {
  vector<int> zero_tensor_strides(D);
  fill(zero_tensor_strides.begin(), zero_tensor_strides.end(), 0);
  uint64_t out_coords_key = hash_vec(zero_tensor_strides);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  coords_maps[out_coords_key] = CoordsMap(D + 1, batch_indices);
  return out_coords_key;
}

const InOutMapKey CoordsManager::getMapHashKey(
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose, bool is_pool) const {

  int D = tensor_strides.size();
  ASSERT(D == tensor_strides.size() and D == strides.size() and
             D == kernel_sizes.size() and D == dilations.size(),
         "Size mismatch. tensor_strides: ", tensor_strides.size(),
         ", strides: ", strides.size(), ", kernel_sizes: ", kernel_sizes.size(),
         ", dilations: ", dilations.size());

  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t stride_hash = hash_vec(strides);
  uint64_t kernel_size_hash = hash_vec(kernel_sizes);
  uint64_t dilation_hash = hash_vec(dilations);
  const InOutMapKey map_key = {
      in_coords_key, out_coords_key,        stride_hash,  kernel_size_hash,
      dilation_hash, (uint64_t)region_type, is_transpose, is_pool};

  return map_key;
}

const InOutMapKey
CoordsManager::getOriginMapHashKey(py::object py_in_coords_key,
                                   py::object py_out_coords_key) const {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  ASSERT(
      p_in_coords_key->key_set and p_out_coords_key->key_set,
      "Key is not set. in_coords_key: ", to_string(p_in_coords_key->getKey()),
      ", out_coords_key: ", to_string(p_out_coords_key->getKey()));

  int D = p_in_coords_key->getDimension();

  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  vector<int> zero_vec(D);
  fill(zero_vec.begin(), zero_vec.end(), 0);
  uint64_t zero_hash = hash_vec(zero_vec);
  const InOutMapKey map_key = {
      in_coords_key, out_coords_key, zero_hash, zero_hash, zero_hash, 0, false,
      true};
  return map_key;
}

/**
 * Entry function for coords map generation and the associated kernel maps.
 */
const InOutMapsRefPair<int> CoordsManager::getInOutMaps(
    const vector<int> &tensor_strides, const vector<int> &strides,
    const vector<int> &kernel_sizes, const vector<int> &dilations,
    int region_type, const at::Tensor &offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose, bool is_pool,
    bool force_creation) {

  int D = tensor_strides.size();
  ASSERT(D == tensor_strides.size() and D == strides.size() and
             D == kernel_sizes.size() and D == dilations.size(),
         "Size mismatch. tensor_strides: ", tensor_strides.size(),
         ", strides: ", strides.size(), ", kernel_sizes: ", kernel_sizes.size(),
         ", dilations: ", dilations.size());

  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  // 1. Create output coordinates if it doesn't exist
  //
  // create a new output coordinates if it is transpose and gen new coords
  if (!p_out_coords_key->key_set || force_creation) {
    if (!is_transpose) {
      out_coords_key = createStridedCoords(
          p_in_coords_key->getKey(), tensor_strides, strides, force_creation);
    } else {
      // out_coords_key = createTransposedOutCoords(
      //     p_in_coords_key->getKey(), tensor_strides, strides, kernel_sizes,
      //     dilations, region_type, offsets, force_creation);
      out_coords_key = createTransposedStridedRegionCoords(
          p_in_coords_key->getKey(), tensor_strides, strides, kernel_sizes,
          dilations, region_type, offsets, force_creation);
    }
    p_out_coords_key->setKey(out_coords_key);
  } else {
    out_coords_key = p_out_coords_key->getKey();
  }

  // 2. Generate kernel map
  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  CoordsMap &in_map = coords_maps[in_coords_key];
  CoordsMap &out_map = coords_maps[out_coords_key];

  // Create kernel maps
  if (!is_transpose) { // NON TRANSPOSE
    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->stride(strides);
    // For non transpose case
    // make a kernel mapping. The kernel will be saved with the map_key.
    if (in_maps.find(map_key) == in_maps.end()) {
      const vector<int> out_tensor_strides = computeOutTensorStride(
          tensor_strides, strides, false /* is_transpose */);

      // Create kernel map using the region if it is not a pooling or if the
      // tensor stride is not equal to the kernel size and the region type is
      // not cubic.
      //
      // TODO: even numbered kernel size to use region_type 0
      if (is_pool && (strides == kernel_sizes)) {

        auto in_out = in_map.stride_map(out_map, out_tensor_strides);
        in_maps[map_key] = move(in_out.first);
        out_maps[map_key] = move(in_out.second);

      } else {
        Region region =
            Region(tensor_strides, kernel_sizes, dilations, region_type,
                   offsets.data<int>(), offsets.size(0));

        auto in_out = in_map.kernel_map(out_map, region);
        in_maps[map_key] = move(in_out.first);
        out_maps[map_key] = move(in_out.second);
      }
    }
    return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));

  } else { // TRANSPOSE

    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->up_stride(strides);

    // Create temporary key for the flipped in/out
    const InOutMapKey tmp_map_key = getMapHashKey(
        tensor_strides, strides, kernel_sizes, dilations, region_type,
        py_out_coords_key, py_in_coords_key, false, is_pool);

    // Check if the temporary key exists and return swapped in/out
    if (in_maps.find(tmp_map_key) != in_maps.end()) {
      return make_pair(ref(out_maps[tmp_map_key]), ref(in_maps[tmp_map_key]));

    } else { // create in out kernel if it doesn't exist

      if (is_pool && strides == kernel_sizes && region_type == 0) {
        // out tensor strides are smaller than in tensor strides for transpose
        auto in_tensor_strides = p_in_coords_key->getTensorStride();
        auto out_in = out_map.stride_map(in_map, in_tensor_strides);

        in_maps[map_key] = move(out_in.second);
        out_maps[map_key] = move(out_in.first);

      } else {
        // out tensor strides are smaller than in tensor strides for transpose
        auto out_tensor_strides = p_out_coords_key->getTensorStride();
        Region region =
            Region(out_tensor_strides, kernel_sizes, dilations, region_type,
                   offsets.data<int>(), offsets.size(0));

        // Flip in and out
        auto out_in = out_map.kernel_map(in_map, region);

        in_maps[map_key] = move(out_in.second);
        out_maps[map_key] = move(out_in.first);
      }

      return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
    }
  }
}

const InOutMapsRefPair<int>
CoordsManager::getOriginInOutMaps(py::object py_in_coords_key,
                                  py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  int D = p_in_coords_key->getDimension();
  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    out_coords_key = createOriginCoords(D);
    p_out_coords_key->setKey(out_coords_key);
    vector<int> zero_vec(D);
    fill(zero_vec.begin(), zero_vec.end(), 0);
    p_out_coords_key->setTensorStride(zero_vec);
  } else
    out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    auto in_out = coords_maps[in_coords_key].global_reduction_map(
        coords_maps[out_coords_key]);
    in_maps[map_key] = in_out.first;
    out_maps[map_key] = in_out.second;
  }
  return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
}

const InOutMapsRefPair<int>
CoordsManager::getPruningInOutMaps(at::Tensor use_feat,
                                   py::object py_in_coords_key,
                                   py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set)
    // The following function setup py_out_coords_key
    out_coords_key =
        createPrunedCoords(use_feat, py_in_coords_key, py_out_coords_key);
  else
    out_coords_key = p_out_coords_key->getKey();

  // Use the map key for origin hash map (stride, dilation, kernel are all NULL)
  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    auto in_out = coords_maps[in_coords_key].pruned_kernel_map(
        coords_maps[out_coords_key]);
    in_maps[map_key] = in_out.first;
    out_maps[map_key] = in_out.second;
  }

  return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
}

string CoordsManager::toString() const {
  string tmp;
  tmp += "< CoordsManager, Number of Coordinate Hashmaps: ";
  tmp += to_string(coords_maps.size());
  for (auto kv : coords_maps) {
    tmp += " \n\tCoords Key: ";
    tmp += to_string(kv.first);
    tmp += ", Size: ";
    tmp += to_string((kv.second).size());
  }
  tmp += "\n  Number of Kernel Maps: ";
  tmp += to_string(in_maps.size());
  for (auto kv : in_maps) {
    tmp += " \n\tKernel Map Key: ";
    tmp += to_string(hash_vec<InOutMapKey>(kv.first));
    int size = 0;
    for (auto map : kv.second) {
      size += map.size();
    }
    tmp += ", Size: ";
    tmp += to_string(size);
  }
  tmp += " >";
  return tmp;
}

void CoordsManager::printDiagnostics(py::object py_coords_key) {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  coords_maps[p_coords_key->getKey()].print();
}

/*
 * Return the batch indices and row indices for each image.
 */
pair<vector<int>, vector<vector<int>>>
CoordsManager::getRowIndicesPerBatch(py::object py_in_coords_key,
                                     py::object py_out_coords_key) {
  // py_out_coords_key will be set after the above call.
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  auto in_coords_key = p_in_coords_key->getKey();
  ASSERT(coords_maps.find(in_coords_key) != coords_maps.end(),
         "The in_coords_key, ", to_string(in_coords_key), ", does not exist.");

  auto in_out = getOriginInOutMaps(py_in_coords_key, py_out_coords_key);

  // list of row indices. The batch index is defined as the returned batch
  // indices
  vector<vector<int>> batch2row_inds(batch_indices.size());
  const auto &in = in_out.first, out = in_out.second;
  for (size_t k = 0; k < in.size(); k++) {
    const auto &curr_in = in[k];
    const auto &curr_out = out[k];
    for (size_t i = 0; i < curr_in.size(); i++)
      batch2row_inds[curr_out[i]].push_back(curr_in[i]);
  }

  // copy batch_indices, move batch2row_inds
  vector<int> vec_batch_indices;
  vec_batch_indices.reserve(batch_indices.size());
  for (int b : batch_indices) {
    vec_batch_indices.push_back(b);
  }

  return make_pair(move(vec_batch_indices), move(batch2row_inds));
}
