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

template <typename T1, typename T2> void copy_types(const T1 &src, T2 &dst) {
  size_t curr_it = 0;
  for (const auto s : src)
    dst[curr_it++] = s;
}

/*
 * Given tensor_stride_src and tensor_stride_dst, find the respective coord_maps
 * and return the indices of the coord_map_ind in coord_map_dst
 */
template <typename MapType>
vector<vector<at::Tensor>> CoordsManager<MapType>::getKernelMap(
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    bool is_transpose, bool is_pool) {
  // WARNING: This function will not work properly with custon region types.
  ASSERT(region_type != 2,
         "Currently, it does not support the custom region type.");
  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  const auto &in_map_iter = in_maps.find(map_key);

  if (in_map_iter == in_maps.end()) {
    getInOutMaps(tensor_strides, strides, kernel_sizes, dilations, region_type,
                 offsets, py_in_coords_key, py_out_coords_key, false);
    ASSERT(in_maps.find(map_key) != in_maps.end(), "Kernel map not found.");
  }

  const InOutMaps<int> &in_map = in_maps[map_key];
  const InOutMaps<int> &out_map = out_maps[map_key];

  int all_volume = 0, kernel_volume = in_map.size();
  for (int k = 0; k < kernel_volume; k++)
    all_volume += in_map[k].size();

  vector<at::Tensor> in_tensors, out_tensors;

  for (int k = 0; k < kernel_volume; k++) {
    auto curr_volume = in_map[k].size();
    if (curr_volume <= 0)
      continue;

    at::Tensor in_kernel_map = torch::empty(
        {(long)curr_volume}, torch::TensorOptions().dtype(torch::kInt64));
    at::Tensor out_kernel_map = torch::empty(
        {(long)curr_volume}, torch::TensorOptions().dtype(torch::kInt64));

    long *p_in_kernel_map = in_kernel_map.data<long>();
    long *p_out_kernel_map = out_kernel_map.data<long>();

    for (int i = 0; i < curr_volume; i++) {
      p_in_kernel_map[i] = in_map[k][i];
      p_out_kernel_map[i] = out_map[k][i];
    }

    in_tensors.push_back(move(in_kernel_map));
    out_tensors.push_back(move(out_kernel_map));
  }

  return {in_tensors, out_tensors};
}

template <typename MapType>
vector<at::Tensor>
CoordsManager<MapType>::getCoordsMap(py::object py_in_coords_key,
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
  const auto in_out =
      in_map_iter->second.stride_map(out_map_iter->second, out_tensor_strides);

  const auto &ins = in_out.first;
  const auto &outs = in_out.second;
  // All size
  const auto N = std::accumulate(ins.begin(), ins.end(), 0,
                                 [](size_t curr_sum, const vector<int> &map) {
                                   return curr_sum + map.size();
                                 });

  at::Tensor in_out_1 =
      torch::empty({N}, torch::TensorOptions().dtype(torch::kInt64));
  at::Tensor in_out_2 =
      torch::empty({N}, torch::TensorOptions().dtype(torch::kInt64));

  auto a_in_out_1 = in_out_1.accessor<long int, 1>();
  auto a_in_out_2 = in_out_2.accessor<long int, 1>();

  size_t curr_it = 0;
  for (const auto &in : ins)
    for (const auto i : in)
      a_in_out_1[curr_it++] = i;

  curr_it = 0;
  for (const auto &out : outs)
    for (const auto o : out)
      a_in_out_2[curr_it++] = o;

  return {in_out_1, in_out_2};
}

// Generate and return the ins -> out map.
template <typename MapType>
pair<vector<at::Tensor>, vector<at::Tensor>>
CoordsManager<MapType>::getUnionMap(vector<py::object> py_in_coords_keys,
                                    py::object py_out_coords_key) {

  // all exception handling will be done inside the following
  const InOutMapsRefPair<int> in_outs =
      getUnionInOutMaps(py_in_coords_keys, py_out_coords_key);
  const auto &ins = in_outs.first;
  const auto &outs = in_outs.second;

  // Size of the in out maps
  const auto N = ins.size();

  // Return torch tensor
  vector<at::Tensor> th_ins;
  vector<at::Tensor> th_outs;
  for (size_t i = 0; i < N; ++i) {
    at::Tensor th_in = torch::empty(
        {(long)ins[i].size()}, torch::TensorOptions().dtype(torch::kInt64));
    at::Tensor th_out = torch::empty(
        {(long)outs[i].size()}, torch::TensorOptions().dtype(torch::kInt64));

    copy_types(ins[i], th_in);
    copy_types(outs[i], th_out);

    th_ins.push_back(move(th_in));
    th_outs.push_back(move(th_out));
  }

  return make_pair(th_ins, th_outs);
}

template <typename MapType>
uint64_t
CoordsManager<MapType>::getCoordsKey(const vector<int> &tensor_strides) const {
  auto tensor_stride_hash = hash_vec<vector<int>>(tensor_strides);
  ASSERT(coords_maps.find(tensor_stride_hash) != coords_maps.end(),
         "The coord map doesn't exist for the given tensor strides ",
         "tensor_stride: ", ArrToString(tensor_strides));
  return tensor_stride_hash;
}

template <typename MapType>
bool CoordsManager<MapType>::existsCoordsKey(const uint64_t coords_key) const {
  return coords_maps.find(coords_key) != coords_maps.end();
}

template <typename MapType>
bool CoordsManager<MapType>::existsCoordsKey(py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  return existsCoordsKey(p_coords_key->getKey());
}

template <typename MapType>
uint64_t CoordsManager<MapType>::getRandomCoordsKey() {
  uint64_t coords_key = random();
  while (coords_maps.find(coords_key) != coords_maps.end())
    coords_key = random();
  return coords_key;
}

template <typename MapType>
int CoordsManager<MapType>::getCoordsSize(const uint64_t coords_key) const {
  const auto &coords_map_iter = coords_maps.find(coords_key);
  ASSERT(coords_map_iter != coords_maps.end(),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");
  return coords_map_iter->second.size();
}

template <typename MapType>
int CoordsManager<MapType>::getCoordsSize(py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  return getCoordsSize(p_coords_key->getKey());
}

template <typename MapType>
void CoordsManager<MapType>::getCoords(at::Tensor coords,
                                       py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  const uint64_t coords_key = p_coords_key->getKey();

  // initialize
  const auto &coords_map_iter = coords_maps.find(coords_key);
  ASSERT(coords_map_iter != coords_maps.end(),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");
  const CoordsMap<MapType> &coordmap = coords_map_iter->second;
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

template <typename MapType>
void CoordsManager<MapType>::setOriginCoordsKey(py::object py_coords_key) {
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
uint64_t CoordsManager<MapType>::initializeCoords(
    at::Tensor coords, at::Tensor mapping, at::Tensor inverse_mapping,
    const vector<int> &tensor_strides, const bool force_creation,
    const bool force_remap, const bool allow_duplicate_coords,
    const bool return_inverse) {
  const int nrows = coords.size(0);
  const int ncols = coords.size(1);
  const int D = ncols - 1;

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
             "For more information, please refer to the SparseTensor creation "
             "documentation available at:"
             "https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html");
    }
  }

  // Create the concurrent coords map
  int *p_coords = coords.data<int>();
  tuple<vector<int>, vector<int>, set<int>> map_inverse_map_batch;
  CoordsMap<MapType> coords_map;
  map_inverse_map_batch = coords_map.initialize_batch(
      p_coords, nrows, ncols, force_remap, return_inverse);

  if (!is_batch_indices_set) {
    batch_indices = std::get<2>(map_inverse_map_batch);
    vec_batch_indices = vector<int>(batch_indices.begin(), batch_indices.end());
    is_batch_indices_set = true;
  }

  if (!allow_duplicate_coords && !force_remap) {
    ASSERT(nrows == coords_map.size(), "Duplicate coordinates found. ",
           "Number of input coords:", nrows,
           " != Number of unique coords:", coords_map.size(),
           "If the duplication was intentional, set force_remap to true."
           "For more information, please refer to the SparseTensor creation "
           "documentation available at: "
           "https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html");
  }

  // When remapping, return the mapping to pytorch.
  if (force_remap || return_inverse) {
    ASSERT(mapping.dtype() == torch::kInt64,
           "Mapping must be a torch::LongTensor");
    const vector<int> &map = std::get<0>(map_inverse_map_batch);
    mapping.resize_({(long)map.size()});
    copy(map.begin(), map.end(), mapping.data<long>());
  }

  if (return_inverse) {
    ASSERT(inverse_mapping.dtype() == torch::kInt64,
           "Inverse Mapping must be a torch::LongTensor");
    const vector<int> &inv_map = std::get<1>(map_inverse_map_batch);
    inverse_mapping.resize_({(long)inv_map.size()});
    copy(inv_map.begin(), inv_map.end(), inverse_mapping.data<long>());
  }

  // Save the returned results
  coords_maps[key] = move(coords_map);

  return key;
}

template <typename MapType>
uint64_t CoordsManager<MapType>::initializeCoords(
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
uint64_t CoordsManager<MapType>::createStridedCoords(
    uint64_t coords_key, const vector<int> &tensor_strides,
    const vector<int> &strides, bool force_creation) {
  // Basic assertions
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");

  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, false);

  const int D = coords_maps[coords_key].ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "CoordsManager dimension: ", to_string(D),
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
    coords_maps[out_coords_key] =
        coords_maps[coords_key].stride(out_tensor_strides);
  }

  return out_coords_key;
}

template <typename MapType>
uint64_t CoordsManager<MapType>::createTransposedStridedRegionCoords(
    uint64_t coords_key, const vector<int> &tensor_strides,
    const vector<int> &strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, at::Tensor offsets, bool force_creation) {
  const vector<int> out_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, true /* is_transpose */);

  // Basic assertions
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(coords_key), ".");

  const int D = coords_maps[coords_key].ncols - 1;
  ASSERT(D == tensor_strides.size(), "The coordinate dimensions mismatch. ",
         "CoordsManager dimension: ", to_string(D),
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

  coords_maps[out_coords_key] = coords_maps[coords_key].stride_region(region);

  return out_coords_key;
}

template <typename MapType>
uint64_t
CoordsManager<MapType>::createPrunedCoords(at::Tensor use_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();

  ASSERT(existsCoordsKey(in_coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         to_string(in_coords_key), ".");

  // set a random coords key
  const uint64_t out_coords_key = getRandomCoordsKey();

  // Set the pycoordskey
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  p_out_coords_key->setKey(out_coords_key);
  if (!p_out_coords_key->tensor_stride_set)
    p_out_coords_key->setTensorStride(p_in_coords_key->getTensorStride());

  coords_maps[out_coords_key] =
      coords_maps[in_coords_key].prune(use_feat.data<bool>(), use_feat.size(0));

  return out_coords_key;
}

template <typename MapType>
uint64_t CoordsManager<MapType>::createOriginCoords(const int D) {
  const vector<int> zero_tensor_strides(D, 0);
  const uint64_t out_coords_key = hash_vec(zero_tensor_strides);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  coords_maps[out_coords_key] = CoordsMap<MapType>(D + 1, batch_indices);
  return out_coords_key;
}

template <typename MapType>
uint64_t
CoordsManager<MapType>::createUnionCoords(vector<py::object> py_in_coords_keys,
                                          py::object py_out_coords_key) {
  vector<CoordsKey *> p_in_coords_keys;
  CoordsKey *p_in_coords_key = py_in_coords_keys[0].cast<CoordsKey *>();
  auto tensor_strides = p_in_coords_key->getTensorStride();
  for (const auto &py_in_coords_key : py_in_coords_keys) {
    // Set the tensor strides to the smallest elements.
    p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
    p_in_coords_keys.push_back(p_in_coords_key);
    transform(tensor_strides.begin(),                            /* In1 begin */
              tensor_strides.end(),                              /* In1 end */
              p_in_coords_key->getTensorStride().begin(),        /* In2 begin */
              tensor_strides.begin(),                            /* out begin */
              [](int a, int b) -> int { return std::min(a, b); } /* binary op */
    );
    const uint64_t in_coords_key = p_in_coords_key->getKey();
    ASSERT(existsCoordsKey(in_coords_key),
           "The coord map doesn't exist for the given coords_key: ",
           to_string(in_coords_key), ".");
  }
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  vector<reference_wrapper<CoordsMap<MapType>>> in_coords_maps;
  for (const CoordsKey *p_in_coords_key : p_in_coords_keys) {
    CoordsMap<MapType> &curr_map = coords_maps[p_in_coords_key->getKey()];
    in_coords_maps.push_back(ref(curr_map));
  }

  // set a random coords key
  const uint64_t out_coords_key = getRandomCoordsKey();

  // Set the pycoordskey using the last coords_key
  p_out_coords_key->setDimension(p_in_coords_key->getDimension());
  p_out_coords_key->setKey(out_coords_key);
  p_out_coords_key->setTensorStride(tensor_strides);

  coords_maps[out_coords_key] =
      CoordsMap<MapType>::union_coords(in_coords_maps);

  return out_coords_key;
}

template <typename MapType>
const InOutMapKey CoordsManager<MapType>::getMapHashKey(
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
const InOutMapKey CoordsManager<MapType>::getOriginMapHashKey(
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
CoordsManager<MapType>::getUnionMapHashKey(vector<py::object> py_in_coords_keys,
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
const InOutMapsRefPair<int> CoordsManager<MapType>::getInOutMaps(
    const vector<int> &tensor_strides, const vector<int> &strides,
    const vector<int> &kernel_sizes, const vector<int> &dilations,
    int region_type, const at::Tensor &offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose, bool is_pool,
    bool force_creation) {
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

  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();
  const uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t out_coords_key;

  // 1. Create output coordinates if it doesn't exist
  //
  // create a new output coordinates if it is transpose and gen new coords
  if (!p_out_coords_key->key_set || force_creation) {
    if (!is_transpose) {
      // Will return the in_coords_key if strides == 1.
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

  CoordsMap<MapType> &in_map = coords_maps[in_coords_key];
  CoordsMap<MapType> &out_map = coords_maps[out_coords_key];

  // Create kernel maps
  if (!is_transpose) { // NON TRANSPOSE
    if (!p_out_coords_key->tensor_stride_set) {
      p_out_coords_key->setTensorStride(tensor_strides);
      p_out_coords_key->stride(strides);
    }
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

        const auto in_out = in_map.stride_map(out_map, out_tensor_strides);
        in_maps[map_key] = move(in_out.first);
        out_maps[map_key] = move(in_out.second);

      } else {
        Region region =
            Region(tensor_strides, kernel_sizes, dilations, region_type,
                   offsets.data<int>(), offsets.size(0));

        const auto in_out = in_map.kernel_map(out_map, region);
        in_maps[map_key] = move(in_out.first);
        out_maps[map_key] = move(in_out.second);
      }
    }
    return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));

  } else { // TRANSPOSE

    if (!p_out_coords_key->tensor_stride_set) {
      p_out_coords_key->setTensorStride(tensor_strides);
      p_out_coords_key->up_stride(strides);
    }

    // Create temporary key for the flipped in/out
    const InOutMapKey tmp_map_key = getMapHashKey(
        tensor_strides, strides, kernel_sizes, dilations, region_type,
        py_out_coords_key, py_in_coords_key, false, is_pool);

    // Check if the temporary key exists and return swapped in/out
    if (in_maps.find(tmp_map_key) != in_maps.end()) {
      // copy the in out maps from the existing maps
      in_maps[map_key] = out_maps[tmp_map_key];
      out_maps[map_key] = in_maps[tmp_map_key];

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
    }
    return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
  }
}

template <typename MapType>
const InOutMapsRefPair<int>
CoordsManager<MapType>::getOriginInOutMaps(py::object py_in_coords_key,
                                           py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  const int D = p_in_coords_key->getDimension();
  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    p_out_coords_key->setKey(createOriginCoords(D));
    const vector<int> zero_vec(D, 0);
    p_out_coords_key->setTensorStride(zero_vec);
  } else {
    ASSERT(coords_maps.find(p_out_coords_key->getKey()) != coords_maps.end(),
           "Global map not initialized.");
  }

  const uint64_t in_coords_key = p_in_coords_key->getKey();
  const uint64_t out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    ASSERT(coords_maps[out_coords_key].size() == batch_indices.size(),
           "Coords size mismatch. CoordsMap size: ",
           coords_maps[out_coords_key].size(),
           ", batch size: ", batch_indices.size());
    const auto in_out = coords_maps[in_coords_key].global_reduction_map(
        coords_maps[out_coords_key]);
    in_maps[map_key] = in_out.first;
    out_maps[map_key] = in_out.second;
  }
  return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
}

template <typename MapType>
const InOutMapsRefPair<int>
CoordsManager<MapType>::getPruningInOutMaps(at::Tensor use_feat,
                                            py::object py_in_coords_key,
                                            py::object py_out_coords_key) {
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    // The following function setup py_out_coords_key
    createPrunedCoords(use_feat, py_in_coords_key, py_out_coords_key);
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
    const auto in_out = coords_maps[in_coords_key].pruned_kernel_map(
        coords_maps[out_coords_key]);
    in_maps[map_key] = in_out.first;
    out_maps[map_key] = in_out.second;
  }

  return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
}

template <typename MapType>
const InOutMapsRefPair<int>
CoordsManager<MapType>::getUnionInOutMaps(vector<py::object> py_in_coords_keys,
                                          py::object py_out_coords_key) {
  CoordsKey *p_out_coords_key = py_out_coords_key.cast<CoordsKey *>();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set)
    createUnionCoords(py_in_coords_keys, py_out_coords_key);

  const uint64_t out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  const InOutMapKey map_key =
      getUnionMapHashKey(py_in_coords_keys, py_out_coords_key);

  vector<reference_wrapper<CoordsMap<MapType>>> in_coords_maps;
  for (const auto &py_in_coords_key : py_in_coords_keys) {
    const CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
    uint64_t in_coords_key = p_in_coords_key->getKey();
    in_coords_maps.push_back(ref(coords_maps[in_coords_key]));
  }

  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    const auto in_out = CoordsMap<MapType>::union_map(
        in_coords_maps, coords_maps[out_coords_key]);
    in_maps[map_key] = in_out.first;
    out_maps[map_key] = in_out.second;
  }

  return make_pair(ref(in_maps[map_key]), ref(out_maps[map_key]));
}

template <typename MapType> string CoordsManager<MapType>::toString() const {
  Formatter out;
  out << "< CoordsManager\n\tNumber of Coordinate Maps: "
      << to_string(coords_maps.size());
  for (const auto &kv : coords_maps) {
    out << " \n\t\tCoordinate Map Key: " << to_string(kv.first)
        << ", Size: " << to_string((kv.second).size());
  }
  out << "\n\tNumber of Kernel Maps: " << to_string(in_maps.size());
  for (const auto &kv : in_maps) {
    size_t size = 0;
    for (const auto &map : kv.second)
      size += map.size();
    out << " \n\t\tKernel In-Out Map Key: "
        << to_string(hash_vec<InOutMapKey>(kv.first))
        << ", Size: " << to_string(size);
  }
  out << " >\n";
  return out;
}

template <typename MapType>
void CoordsManager<MapType>::printDiagnostics(py::object py_coords_key) const {
  CoordsKey *p_coords_key = py_coords_key.cast<CoordsKey *>();
  const auto &map_iter = coords_maps.find(p_coords_key->getKey());
  ASSERT(map_iter != coords_maps.end(), "Coords map does not exist.");
  map_iter->second.print();
}

/*
 * Return row indices for each batch index
 */
template <typename MapType>
at::Tensor
CoordsManager<MapType>::getRowIndicesAtBatchIndex(py::object py_in_coords_key,
                                                  py::object py_out_coords_key,
                                                  const int batch_index) {
  // py_out_coords_key will be set after the above call.
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  const auto in_coords_key = p_in_coords_key->getKey();
  ASSERT(coords_maps.find(in_coords_key) != coords_maps.end(),
         "The in_coords_key, ", to_string(in_coords_key), ", does not exist.");
  // Find the batch index in the current batch indices.
  const vector<int>::iterator batch_iter = std::find(
      vec_batch_indices.begin(), vec_batch_indices.end(), batch_index);

  // ASSERT(batch_iter != vec_batch_indices.end(),
  //        "Invalid batch index:", batch_index,
  //        " does not exist in the provided batch indices:",
  //        ArrToString(vec_batch_indices));

  // Return an empty list if not found.
  if (batch_iter == vec_batch_indices.end()) {
    at::Tensor in_rows =
        torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64));
    return in_rows;

  } else {

    const auto in_outs =
        getOriginInOutMaps(py_in_coords_key, py_out_coords_key);
    const auto &in = in_outs.first[*batch_iter];

    at::Tensor in_rows = torch::zeros(
        {(long)in.size()}, torch::TensorOptions().dtype(torch::kInt64));

    // copy all from a vector. int -> long
    auto a_in_rows = in_rows.accessor<long, 1>();
    for (auto i = 0; i < in.size(); i++)
      a_in_rows[i] = in[i];

    return in_rows;
  }
}

/*
 * Return row indices per batch
 */
template <typename MapType>
vector<at::Tensor>
CoordsManager<MapType>::getRowIndicesPerBatch(py::object py_in_coords_key,
                                              py::object py_out_coords_key) {
  // py_out_coords_key will be set after the above call.
  CoordsKey *p_in_coords_key = py_in_coords_key.cast<CoordsKey *>();
  const auto in_coords_key = p_in_coords_key->getKey();
  ASSERT(coords_maps.find(in_coords_key) != coords_maps.end(),
         "The in_coords_key, ", to_string(in_coords_key), ", does not exist.");

  const auto in_outs = getOriginInOutMaps(py_in_coords_key, py_out_coords_key);
  const auto &ins = in_outs.first;

  // Return index.
  vector<at::Tensor> out_inds;
  for (const auto &in : ins) {
    at::Tensor mapping = torch::zeros(
        {(long)in.size()}, torch::TensorOptions().dtype(torch::kInt64));

    // copy all from a vector, int -> long
    auto a_mapping = mapping.accessor<long, 1>();
    for (auto i = 0; i < in.size(); i++)
      a_mapping[i] = in[i];

    // ::memcpy(mapping.data<int>(), in.data(), in.size() * sizeof(int));
    out_inds.push_back(mapping);
  }

  return out_inds;
}

template class CoordsManager<CoordsToIndexMap>;
// template class CoordsManager<CoordsToVectorMap>;

} // end namespace minkowski
