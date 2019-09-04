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
#include "coords_hashmaps.hpp"
#include "coords_kernelmaps.hpp"
#include "region_iter.hpp"
#include "utils.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

std::vector<int> computeOutTensorStride(const std::vector<int> &tensor_strides,
                                        const std::vector<int> &strides,
                                        bool is_transpose) {
  std::vector<int> out_tensor_strides;
  ASSERT(tensor_strides.size() == strides.size(),
         "The dimension of tensor_stride: ", ArrToString(tensor_strides),
         " does not match the dimension of strides: ", ArrToString(strides));
  for (int i = 0; i < strides.size(); i++) {
    if (is_transpose) {
      ASSERT(tensor_strides[i] % strides[i] == 0,
             "The output tensor stride is not divisible by ",
             "up_strides. tensor stride: ", ArrToString(tensor_strides),
             ", up_strides: ", ArrToString(strides));
      out_tensor_strides.push_back(tensor_strides[i] / strides[i]);
    } else
      out_tensor_strides.push_back(tensor_strides[i] * strides[i]);
  }
  return out_tensor_strides;
}

// TODO
template <typename Itype> CoordsManager<Itype>::CoordsManager() {}

/*
 * Given tensor_stride_src and tensor_stride_dst, find the respective coord_maps
 * and return the indices of the coord_map_ind in coord_map_dst
 */
template <typename Itype>
void CoordsManager<Itype>::getKernelMap(
    at::Tensor kernel_map, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose) {
  InOutMapKey map_key = getMapHashKey(tensor_strides, strides, kernel_sizes,
                                      dilations, region_type, py_in_coords_key,
                                      py_out_coords_key, is_transpose);

  if (_in_maps.find(map_key) == _in_maps.end()) {
    throw std::invalid_argument(Formatter() << "The kernelmap does not exist.");
  }

  const InOutMapPerKernel<Itype> &in_map = _in_maps[map_key];
  const InOutMapPerKernel<Itype> &out_map = _out_maps[map_key];

  int all_volume = 0, kernel_volume = in_map.size();
  for (int k = 0; k < kernel_volume; k++)
    all_volume += in_map[k].size();

  kernel_map.resize_({all_volume, 3});
  Itype *p_kernel_map = kernel_map.data<Itype>();

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

template <typename Itype>
uint64_t
CoordsManager<Itype>::getCoordsKey(const std::vector<int> &tensor_strides) {
  auto tensor_stride_hash = hash_vec<std::vector<int>>(tensor_strides);
  uint64_t r_coords_key = 0;

  // Following lines are from INITIALIZE_IN_COORDS
  /* Prioritize the p_coords_key */
  if (_coords_hashmaps.find(tensor_stride_hash) != _coords_hashmaps.end()) {
    r_coords_key = tensor_stride_hash;
  } else {
    throw std::invalid_argument(
        Formatter()
        << "The coord map doesn't exist for the given tensor strides "
        << "tensor_stride: " << ArrToString(tensor_strides) << " at "
        << __FILE__ << ":" << __LINE__);
  }

  return r_coords_key;
}

template <typename Itype>
bool CoordsManager<Itype>::existsCoordsKey(uint64_t coords_key) {
  bool exist = false;
  // Following lines are from INITIALIZE_IN_COORDS
  /* Prioritize the p_coords_key */
  if (_coords_hashmaps.find(coords_key) != _coords_hashmaps.end())
    exist = true;
  return exist;
}

template <typename Itype>
bool CoordsManager<Itype>::existsCoordsKey(py::object py_coords_key) {
  PyCoordsKey *p_coords_key = py_coords_key.cast<PyCoordsKey *>();
  return existsCoordsKey(p_coords_key->getKey());
}

template <typename Itype>
int CoordsManager<Itype>::getCoordsSize(uint64_t coords_key) {
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         std::to_string(coords_key), ".");
  return _coords_hashmaps[coords_key].size();
}

template <typename Itype>
int CoordsManager<Itype>::getCoordsSize(py::object py_coords_key) {
  PyCoordsKey *p_coords_key = py_coords_key.cast<PyCoordsKey *>();
  return getCoordsSize(p_coords_key->getKey());
}

template <typename Itype>
void CoordsManager<Itype>::getCoords(at::Tensor coords,
                                     py::object py_coords_key) {
  PyCoordsKey *p_coords_key = py_coords_key.cast<PyCoordsKey *>();
  uint64_t coords_key = p_coords_key->getKey();
  int nrows = getCoordsSize(coords_key);
  // initialize
  int D = p_coords_key->getDimension();
  coords.resize_({nrows, D + 1});
  // copy to the out coords
  Itype *p_coords = coords.data<Itype>();
  auto &curr_coords_pair = _coords_pairs[coords_key];
  ASSERT(curr_coords_pair.first == D + 1,
         "The coordinate dimensions mismatch. ",
         "CoordsManager dimension: ", std::to_string(curr_coords_pair.first),
         ", PyCoordsKey dimension: ", std::to_string(D));
  auto &curr_coords = curr_coords_pair.second;
  std::copy(curr_coords.begin(), curr_coords.end(), p_coords);
}

/*******************************
 * Initialization
 *******************************/
template <typename Itype>
uint64_t
CoordsManager<Itype>::initializeCoords(at::Tensor coords,
                                       const std::vector<int> &tensor_strides,
                                       bool enforce_creation) {
  int ncols = coords.size(1);
  int D = ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimension (ncols - 1) ",
         std::to_string(D),
         " must match the size of tensor stride: ", ArrToString(tensor_strides),
         ".");

  uint64_t key = hash_vec(tensor_strides);
  bool key_exists = _coords_hashmaps.find(key) != _coords_hashmaps.end();
  if (key_exists) {
    ASSERT(enforce_creation,
           "The coord map already exists for the given tensor stride ",
           "tensor_stride: ", ArrToString(tensor_strides),
           ". To enforce creation, put true on the enforce_creation.");
    key = random();
    while (_coords_hashmaps.find(key) != _coords_hashmaps.end())
      key = random();
  } // If key doesn't exist, use the current key regardless of enforce creation
  auto coords_batch_vector = createCoordsHashMap(coords);

  // Save the returned results
  _coords_hashmaps[key] = std::move(std::get<0>(coords_batch_vector));
  auto &set_batch_indices = std::get<1>(coords_batch_vector);
  _coords_pairs[key] =
      std::make_pair(D + 1, std::move(std::get<2>(coords_batch_vector)));

  // Initialize batch indices
  _batch_indices.resize(set_batch_indices.size());
  std::copy(set_batch_indices.begin(), set_batch_indices.end(),
            _batch_indices.begin());
  return key;
}

template <typename Itype>
uint64_t CoordsManager<Itype>::initializeCoords(at::Tensor coords,
                                                py::object py_coords_key,
                                                bool enforce_creation) {
  PyCoordsKey *p_coords_key = py_coords_key.cast<PyCoordsKey *>();
  uint64_t in_coords_key =
      initializeCoords(coords, p_coords_key->tensor_strides_, enforce_creation);
  // Assume the dimension is coords.size(1) - 1 (for batch index).
  int D = coords.size(1) - 1;
  ASSERT(D > 0, "Input coordinate dimension, ", std::to_string(D + 1),
         ", is not supported.");
  std::vector<int> tensor_stride(D);
  std::fill(tensor_stride.begin(), tensor_stride.end(), 1);
  p_coords_key->setDimension(coords.size(1) - 1);
  // Tensor strided initialized in python side.
  // p_coords_key->setTensorStride(tensor_stride);
  p_coords_key->setKey(in_coords_key);
  return in_coords_key;
}

/*********************************/
template <typename Itype>
uint64_t CoordsManager<Itype>::createOutCoords(
    uint64_t coords_key, const std::vector<int> &tensor_strides,
    const std::vector<int> &strides, bool is_transpose) {
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         std::to_string(coords_key), ".");

  auto out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, is_transpose);

  int D = _coords_pairs[coords_key].first - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "CoordsManager dimension: ", std::to_string(D),
         ", tensor_strides dimension: ", std::to_string(tensor_strides.size()));
  // tensor_strides.size() == strides.size() on computeOutTensorStride
  uint64_t out_coords_key = hash_vec(out_tensor_strides);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  // Only computes when the strides are non identity
  bool is_identity = true;
  for (auto s : strides)
    if (s != 1)
      is_identity = false;

  if (!is_identity) {
    auto out_pair =
        createOutCoordsHashCoordsPair(coords_key, tensor_strides, strides);
    _coords_hashmaps[out_coords_key] = std::move(out_pair.first);
    _coords_pairs[out_coords_key] =
        std::make_pair(D + 1, std::move(out_pair.second));
  }
  return out_coords_key;
}

template <typename Itype>
uint64_t CoordsManager<Itype>::createPruneCoords(at::Tensor use_feat,
                                                 py::object py_in_coords_key,
                                                 py::object py_out_coords_key) {
  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  // set a random coords key
  uint64_t out_coords_key = random();
  while (_coords_hashmaps.find(out_coords_key) != _coords_hashmaps.end())
    out_coords_key = random();
  // Set the pycoordskey
  p_out_coords_key->setTensorStride(p_in_coords_key->getTensorStride());
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  p_out_coords_key->setKey(out_coords_key);

  int D = p_in_coords_key->getDimension();

  // Create coords hashmap
  auto coords_pair =
      createPrunedCoordsHashMap(p_in_coords_key->getKey(), use_feat);
  _coords_hashmaps[out_coords_key] = std::move(coords_pair.first);
  _coords_pairs[out_coords_key] =
      std::make_pair(D + 1, std::move(coords_pair.second));

  return out_coords_key;
}

template <typename Itype>
uint64_t CoordsManager<Itype>::createOriginCoords(int D) {
  std::vector<int> zero_tensor_strides(D);
  std::fill(zero_tensor_strides.begin(), zero_tensor_strides.end(), 0);
  uint64_t out_coords_key = hash_vec(zero_tensor_strides);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  auto coords_pair = createOriginCoordsHashMap(D);
  _coords_hashmaps[out_coords_key] = std::move(coords_pair.first);
  _coords_pairs[out_coords_key] =
      std::make_pair(D + 1, std::move(coords_pair.second));

  return out_coords_key;
}

template <typename Itype>
InOutMapKey CoordsManager<Itype>::getMapHashKey(
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    bool is_transpose) {

  int D = tensor_strides.size();
  ASSERT(D == tensor_strides.size() and D == strides.size() and
             D == kernel_sizes.size() and D == dilations.size(),
         "Size mismatch. tensor_strides: ", tensor_strides.size(),
         ", strides: ", strides.size(), ", kernel_sizes: ", kernel_sizes.size(),
         ", dilations: ", dilations.size());

  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t stride_hash = hash_vec(strides);
  uint64_t kernel_size_hash = hash_vec(kernel_sizes);
  uint64_t dilation_hash = hash_vec(dilations);
  InOutMapKey map_key = {
      in_coords_key, out_coords_key,        stride_hash, kernel_size_hash,
      dilation_hash, (uint64_t)region_type, is_transpose};
  return map_key;
}

template <typename Itype>
InOutMapKey
CoordsManager<Itype>::getOriginMapHashKey(py::object py_in_coords_key,
                                          py::object py_out_coords_key) {
  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  int D = p_in_coords_key->getDimension();

  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  std::vector<int> zero_vec(D);
  std::fill(zero_vec.begin(), zero_vec.end(), 0);
  uint64_t zero_hash = hash_vec(zero_vec);
  InOutMapKey map_key = {
      in_coords_key, out_coords_key, zero_hash, zero_hash, zero_hash, 0, false};
  return map_key;
}

template <typename Itype>
InOutMapKey
CoordsManager<Itype>::getOriginMapHashKeyCheck(py::object py_in_coords_key,
                                               py::object py_out_coords_key) {
  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  ASSERT(p_in_coords_key->key_set and p_out_coords_key->key_set,
         "Key is not set. in_coords_key: ",
         std::to_string(p_in_coords_key->getKey()),
         ", out_coords_key: ", std::to_string(p_out_coords_key->getKey()));
  // Use the global pooling mapping
  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  std::vector<int> zero_vec(p_in_coords_key->getDimension());
  std::fill(zero_vec.begin(), zero_vec.end(), 0);
  uint64_t zero_hash = hash_vec(zero_vec);
  InOutMapKey map_key = {
      in_coords_key, out_coords_key, zero_hash, zero_hash, zero_hash, 0, false};
  return map_key;
}

template <typename Itype>
std::pair<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
CoordsManager<Itype>::setupAndReturnInOutPerKernel(
    const std::vector<int> &tensor_strides, const std::vector<int> &strides,
    const std::vector<int> &kernel_sizes, const std::vector<int> &dilations,
    int region_type, const at::Tensor &offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose) {

  int D = tensor_strides.size();
  ASSERT(D == tensor_strides.size() and D == strides.size() and
             D == kernel_sizes.size() and D == dilations.size(),
         "Size mismatch. tensor_strides: ", tensor_strides.size(),
         ", strides: ", strides.size(), ", kernel_sizes: ", kernel_sizes.size(),
         ", dilations: ", dilations.size());

  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    out_coords_key = createOutCoords(p_in_coords_key->getKey(), tensor_strides,
                                     strides, is_transpose);
    p_out_coords_key->setKey(out_coords_key);
  } else
    out_coords_key = p_out_coords_key->getKey();

  InOutMapKey map_key = getMapHashKey(tensor_strides, strides, kernel_sizes,
                                      dilations, region_type, py_in_coords_key,
                                      py_out_coords_key, is_transpose);

  if (!is_transpose) { // NON TRANSPOSE
    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->stride(strides);
    // For non transpose case
    // make a kernel mapping. The kernel will be saved with the map_key.
    if (_in_maps.find(map_key) == _in_maps.end()) {
      auto in_out =
          createInOutPerKernel(in_coords_key, out_coords_key, tensor_strides,
                               kernel_sizes, dilations, region_type, offsets);
      _in_maps[map_key] = std::get<0>(in_out);
      _out_maps[map_key] = std::get<1>(in_out);
    }
    return std::make_pair(std::ref(_in_maps[map_key]),
                          std::ref(_out_maps[map_key]));

  } else { // TRANSPOSE
    p_out_coords_key->setTensorStride(tensor_strides);
    p_out_coords_key->up_stride(strides);
    // Create temporary key for the flipped in/out
    InOutMapKey tmp_map_key =
        getMapHashKey(tensor_strides, strides, kernel_sizes, dilations,
                      region_type, py_out_coords_key, py_in_coords_key, false);
    // Check if the temporary key exists and return swapped in/out
    if (_in_maps.find(tmp_map_key) != _in_maps.end()) {
      return std::make_pair(std::ref(_out_maps[tmp_map_key]),
                            std::ref(_in_maps[tmp_map_key]));
    } else {
      // create in out kernel if it doesn't exist
      auto out_tensor_strides = p_out_coords_key->getTensorStride();
      auto in_out = createInOutPerKernelTranspose(
          in_coords_key, out_coords_key, out_tensor_strides, kernel_sizes,
          dilations, region_type, offsets);
      _in_maps[map_key] = std::get<0>(in_out);
      _out_maps[map_key] = std::get<1>(in_out);

      return std::make_pair(std::ref(_in_maps[map_key]),
                            std::ref(_out_maps[map_key]));
    }
  }
}

template <typename Itype>
std::pair<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
CoordsManager<Itype>::setupAndReturnOriginInOutPerKernel(
    py::object py_in_coords_key, py::object py_out_coords_key) {
  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  int D = p_in_coords_key->getDimension();
  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    out_coords_key = createOriginCoords(D);
    p_out_coords_key->setKey(out_coords_key);
    std::vector<int> zero_vec(D);
    std::fill(zero_vec.begin(), zero_vec.end(), 0);
    p_out_coords_key->setTensorStride(zero_vec);
  } else
    out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);
  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (_in_maps.find(map_key) == _in_maps.end()) {
    auto in_out = createGlobalReductionInOutMap(in_coords_key, out_coords_key);
    _in_maps[map_key] = std::get<0>(in_out);
    _out_maps[map_key] = std::get<1>(in_out);
  }
  return std::make_pair(std::ref(_in_maps[map_key]),
                        std::ref(_out_maps[map_key]));
}

template <typename Itype>
std::pair<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
CoordsManager<Itype>::setupAndReturnPruningInOutPerKernel(
    at::Tensor use_feat, py::object py_in_coords_key,
    py::object py_out_coords_key) {
  PyCoordsKey *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey *>();
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set)
    // The following function setup py_out_coords_key
    out_coords_key =
        createPruneCoords(use_feat, py_in_coords_key, py_out_coords_key);
  else
    out_coords_key = p_out_coords_key->getKey();

  // Use the map key for origin hash map (stride, dilation, kernel are all NULL)
  InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (_in_maps.find(map_key) == _in_maps.end()) {
    auto in_out = createPruningInOutMap(in_coords_key, out_coords_key);
    _in_maps[map_key] = std::get<0>(in_out);
    _out_maps[map_key] = std::get<1>(in_out);
  }

  return std::make_pair(std::ref(_in_maps[map_key]),
                        std::ref(_out_maps[map_key]));
}

template <typename Itype> std::string CoordsManager<Itype>::toString() const {
  std::string tmp;
  tmp += "< CoordsManager, Number of Coordinate Hashmaps: ";
  tmp += std::to_string(_coords_hashmaps.size());
  for (auto kv : _coords_hashmaps) {
    tmp += " \n\tCoords Key: ";
    tmp += std::to_string(kv.first);
    tmp += ", Size: ";
    tmp += std::to_string((kv.second).size());
  }
  tmp += "\n  Number of Kernel Maps: ";
  tmp += std::to_string(_in_maps.size());
  for (auto kv : _in_maps) {
    tmp += " \n\tKernel Map Key: ";
    tmp += std::to_string(hash_vec<InOutMapKey>(kv.first));
    int size = 0;
    for (auto map : kv.second) {
      size += map.size();
    }
    tmp += ", Size: ";
    tmp += std::to_string(size);
  }
  tmp += " >";
  return tmp;
}

/*
 * Return the batch indices and row indices for each image.
 */
template <typename Itype>
std::pair<std::vector<Itype>, std::vector<std::vector<Itype>>>
CoordsManager<Itype>::getRowIndicesPerBatch(py::object py_in_coords_key,
                                            py::object py_out_coords_key) {
  auto in_out =
      setupAndReturnOriginInOutPerKernel(py_in_coords_key, py_out_coords_key);
  // py_out_coords_key will be set after the above call.
  PyCoordsKey *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey *>();
  auto out_coords_key = p_out_coords_key->getKey();
  auto out_coords_iter = _coords_hashmaps.find(out_coords_key);
  if (out_coords_iter == _coords_hashmaps.end())
    throw std::invalid_argument(Formatter()
                                << "The out_coords_key, " << out_coords_key
                                << ", does not exist.");

  // list of row indices. The batch index is defined as the returned batch
  // indices
  std::vector<std::vector<Itype>> batch2row_inds(_batch_indices.size());
  const auto &in = std::get<0>(in_out), out = std::get<1>(in_out);
  for (std::size_t k = 0; k < in.size(); k++) {
    const auto &curr_in = in[k];
    const auto &curr_out = out[k];
    for (std::size_t i = 0; i < curr_in.size(); i++)
      batch2row_inds[curr_out[i]].push_back(curr_in[i]);
  }

  // copy batch_indices, move batch2row_inds
  return std::make_pair(_batch_indices, std::move(batch2row_inds));
}

template class CoordsManager<int32_t>;
