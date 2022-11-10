/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */

#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "coordinate_map_cpu.hpp"

// #ifndef CPU_ONLY
// #include <ATen/cuda/CUDAContext.h>
// #endif
#include "utils.hpp"

namespace py = pybind11;

namespace minkowski {

/*
struct IndexLabel {
  int index;
  int label;

  IndexLabel() : index(-1), label(-1) {}
  IndexLabel(int index_, int label_) : index(index_), label(label_) {}
};

using cpu_map_type =
    robin_hood::unordered_flat_map<std::vector<int>, int,
                                   byte_hash_vec<int>>;
*/

std::vector<py::array> quantize_np(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> coords) {
  using coordinate_type = int32_t;
  LOG_DEBUG("quantize_np");
  py::buffer_info coords_info = coords.request();
  LOG_DEBUG("buffer info requenst");
  auto &shape = coords_info.shape;

  ASSERT(shape.size() == 2,
         "Dimension must be 2. The dimension of the input: ", shape.size());

  coordinate_type *p_coords = (coordinate_type *)coords_info.ptr;
  LOG_DEBUG("ptr requenst");
  int nrows = shape[0], ncols = shape[1];

  // Create coords map
  LOG_DEBUG("coordinate map generation");
  std::vector<default_types::size_type> tensor_stride(ncols - 1);
  std::for_each(tensor_stride.begin(), tensor_stride.end(),
                [](auto &i) { i = 1; });

  CoordinateMapCPU<coordinate_type> map(nrows, ncols, tensor_stride);
  LOG_DEBUG("Map nrows:", nrows, "ncols:", ncols);
  auto results = map.insert_and_map<true>(p_coords, p_coords + nrows * ncols);
  LOG_DEBUG("insertion finished");
  auto &mapping = std::get<0>(results);
  auto &inverse_mapping = std::get<1>(results);

  // Copy the concurrent vector to std vector
  py::array_t<int32_t> py_mapping = py::array_t<int32_t>(mapping.size());
  py::array_t<int32_t> py_inverse_mapping =
      py::array_t<int32_t>(inverse_mapping.size());

  py::buffer_info py_mapping_info = py_mapping.request();
  py::buffer_info py_inverse_mapping_info = py_inverse_mapping.request();
  int32_t *p_py_mapping = (int32_t *)py_mapping_info.ptr;
  int32_t *p_py_inverse_mapping = (int32_t *)py_inverse_mapping_info.ptr;

  std::copy_n(mapping.data(), mapping.size(), p_py_mapping);
  std::copy_n(inverse_mapping.data(), inverse_mapping.size(),
              p_py_inverse_mapping);

  // mapping is empty when coords are all unique
  return {py_mapping, py_inverse_mapping};
}

std::vector<at::Tensor> quantize_th(at::Tensor &coords) {
  using coordinate_type = int32_t;
  ASSERT(coords.dtype() == torch::kInt32,
         "Coordinates must be an int type tensor.");
  ASSERT(coords.dim() == 2,
         "Coordinates must be represnted as a matrix. Dimensions: ",
         coords.dim(), "!= 2.");
  coordinate_type *p_coords = coords.template data_ptr<coordinate_type>();
  size_t nrows = coords.size(0), ncols = coords.size(1);
  std::vector<default_types::size_type> tensor_stride(ncols - 1);
  std::for_each(tensor_stride.begin(), tensor_stride.end(),
                [](auto &i) { i = 1; });

  CoordinateMapCPU<coordinate_type> map(nrows, ncols, tensor_stride);

  auto results = map.insert_and_map<true>(p_coords, p_coords + nrows * ncols);
  auto mapping = std::get<0>(results);
  auto inverse_mapping = std::get<1>(results);

  // Long tensor for for easier indexing
  auto th_mapping = torch::empty({(int64_t)mapping.size()},
                                 torch::TensorOptions().dtype(torch::kInt64));
  auto th_inverse_mapping =
      torch::empty({(int64_t)inverse_mapping.size()},
                   torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_mapping = th_mapping.accessor<int64_t, 1>();
  auto a_th_inverse_mapping = th_inverse_mapping.accessor<int64_t, 1>();

  // Copy the output
  for (size_t i = 0; i < mapping.size(); ++i)
    a_th_mapping[i] = mapping[i];
  for (size_t i = 0; i < inverse_mapping.size(); ++i)
    a_th_inverse_mapping[i] = inverse_mapping[i];

  // mapping is empty when coords are all unique
  return {th_mapping, th_inverse_mapping};
}

std::vector<std::vector<int>> quantize_label(int const *const p_coords,
                                             int const *const p_labels,
                                             int const nrows, int const ncols,
                                             int const invalid_label) {
  // Create coords map
  LOG_DEBUG("coordinate map generation");
  std::vector<default_types::size_type> tensor_stride(ncols - 1);
  std::for_each(tensor_stride.begin(), tensor_stride.end(),
                [](auto &i) { i = 1; });

  // Create coords map
  using coordinate_type = int32_t;
  using key_type = coordinate<coordinate_type>;
  using mapped_type = std::pair<int, int>; // row index and label
  using hasher = detail::coordinate_murmur3<coordinate_type>;
  using key_equal = detail::coordinate_equal_to<coordinate_type>;
  using map_type = robin_hood::unordered_flat_map<key_type,    // key
                                                  mapped_type, // mapped_type
                                                  hasher,      // hasher
                                                  key_equal    // equality
                                                  >;
  using value_type = map_type::value_type;

  auto map = map_type{(size_t)nrows, hasher{(uint32_t)ncols},
                      key_equal{(size_t)ncols}};

  LOG_DEBUG("Map nrows:", nrows, "ncols:", ncols);
  // insert_row
  std::vector<int> mapping;  // N unique
  std::vector<int> colabels; // N unique
  mapping.reserve(nrows);
  colabels.reserve(nrows);
  std::vector<int> inverse_mapping(nrows); // N rows
  int n_unique{0};
  for (int row = 0; row < nrows; ++row) {
    auto key = coordinate<coordinate_type>(p_coords + ncols * row);
    const auto it = map.find(key);
    auto iter_success =
        map.insert(value_type(key, mapped_type(n_unique, p_labels[row])));
    if (iter_success.second) {
      // success
      mapping.push_back(row);
      colabels.push_back(p_labels[row]);
      inverse_mapping[row] = n_unique++;
    } else {
      auto &keyval = *(iter_success.first);
      auto &val = keyval.second;
      // Set the label
      if (val.second != p_labels[row] && val.second != invalid_label) {
        // When the labels differ
        val.second = invalid_label;
        colabels[inverse_mapping[val.first]] = invalid_label;
      }
      inverse_mapping[row] = val.first; // row
    }
  }

  return {mapping, inverse_mapping, colabels};
}

std::vector<py::array> quantize_label_np(
    py::array_t<int, py::array::c_style | py::array::forcecast> coords,
    py::array_t<int, py::array::c_style | py::array::forcecast> labels,
    int invalid_label) {
  py::buffer_info coords_info = coords.request();
  py::buffer_info labels_info = labels.request();
  auto &shape = coords_info.shape;
  auto &lshape = labels_info.shape;

  ASSERT(shape.size() == 2,
         "Dimension must be 2. The dimension of the input: ", shape.size());

  ASSERT(shape[0] == lshape[0], "Coords nrows must be equal to label size.");

  int *p_coords = (int *)coords_info.ptr;
  int *p_labels = (int *)labels_info.ptr;
  int nrows = shape[0], ncols = shape[1];

  auto const &results =
      quantize_label(p_coords, p_labels, nrows, ncols, invalid_label);
  auto const &mapping = results[0];
  auto const &inverse_mapping = results[1];
  auto const &colabels = results[2];

  // Copy the concurrent vector to std vector
  py::array_t<int32_t> py_mapping = py::array_t<int32_t>(mapping.size());
  py::array_t<int32_t> py_inverse_mapping =
      py::array_t<int32_t>(inverse_mapping.size());
  py::array_t<int32_t> py_colabel = py::array_t<int32_t>(colabels.size());

  py::buffer_info py_mapping_info = py_mapping.request();
  py::buffer_info py_inverse_mapping_info = py_inverse_mapping.request();
  py::buffer_info py_colabel_info = py_colabel.request();

  int32_t *p_py_mapping = (int32_t *)py_mapping_info.ptr;
  int32_t *p_py_inverse_mapping = (int32_t *)py_inverse_mapping_info.ptr;
  int32_t *p_py_colabel = (int32_t *)py_colabel_info.ptr;

  std::copy_n(mapping.data(), mapping.size(), p_py_mapping);
  std::copy_n(colabels.data(), colabels.size(), p_py_colabel);
  std::copy_n(inverse_mapping.data(), inverse_mapping.size(),
              p_py_inverse_mapping);

  // mapping is empty when coords are all unique
  return {py_mapping, py_inverse_mapping, py_colabel};
}

std::vector<at::Tensor> quantize_label_th(at::Tensor coords, at::Tensor labels,
                                          int invalid_label) {
  ASSERT(coords.dtype() == torch::kInt32,
         "Coordinates must be an int type tensor.");
  ASSERT(labels.dtype() == torch::kInt32, "Labels must be an int type tensor.");
  ASSERT(coords.dim() == 2,
         "Coordinates must be represnted as a matrix. Dimensions: ",
         coords.dim(), "!= 2.");
  ASSERT(coords.size(0) == labels.size(0),
         "Coords nrows must be equal to label size.");

  int *p_coords = coords.data_ptr<int>();
  int *p_labels = labels.data_ptr<int>();
  int nrows = coords.size(0), ncols = coords.size(1);

  auto const &results =
      quantize_label(p_coords, p_labels, nrows, ncols, invalid_label);
  auto const &mapping = results[0];
  auto const &inverse_mapping = results[1];
  auto const &colabels = results[2];

  // Copy the concurrent vector to std vector
  //
  // Long tensor for for easier indexing
  auto th_mapping = torch::empty({(int64_t)mapping.size()},
                                 torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_mapping = th_mapping.accessor<int64_t, 1>();

  auto th_inverse_mapping =
      torch::empty({(int64_t)inverse_mapping.size()},
                   torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_inverse_mapping = th_inverse_mapping.accessor<int64_t, 1>();

  auto th_colabels = torch::empty({(int64_t)colabels.size()},
                                  torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_colabels = th_colabels.accessor<int64_t, 1>();

  // Copy the output
  for (size_t i = 0; i < mapping.size(); ++i)
    a_th_mapping[i] = mapping[i];
  for (size_t i = 0; i < inverse_mapping.size(); ++i)
    a_th_inverse_mapping[i] = inverse_mapping[i];
  for (size_t i = 0; i < colabels.size(); ++i)
    a_th_colabels[i] = colabels[i];

  return {th_mapping, th_inverse_mapping, th_colabels};
}

/**
 * A collection of feature averaging methods
 * mode == 0: non-weighted average
 * mode == 1: non-weighted sum
 * mode == k: TODO
 *
 * in_feat[in_map[i], j] --> out_feat[out_map[i], j]
 */
// at::Tensor quantization_average_features(
//     at::Tensor th_in_feat /* feature matrix */,
//     at::Tensor th_in_map /* inverse_map from the quantization functions */,
//     at::Tensor th_out_map /* range(N) */, int out_nrows,
//     int mode /* average types */) {
/*
  ASSERT(th_in_feat.dim() == 2, " The feature tensor should be a matrix.");
  ASSERT(th_in_feat.size(0) == th_in_map.size(0),
         "The size of the input feature and the input map must match.");
  ASSERT(th_in_feat.size(0) == th_out_map.size(0),
         "The size of the input map and the output map must match.");
  auto nchannel = th_in_feat.size(1);
  at::Tensor th_out_feat =
      torch::zeros({out_nrows, nchannel}, th_in_feat.options());

  at::Tensor th_num_nonzero = torch::zeros(
      {out_nrows}, torch::TensorOptions().dtype(th_in_feat.dtype()));

#ifndef CPU_ONLY
  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());
#endif

  if (th_in_map.dtype() == torch::kInt64) {
    if (th_in_feat.is_cuda()) {
#ifndef CPU_ONLY
      auto vec_in_map = CopyToInOutMapGPU<int>(th_in_map);
      auto vec_out_map = CopyToInOutMapGPU<int>(th_out_map);

      if (th_in_feat.dtype() == torch::kFloat32) {
        NonzeroAvgPoolingForwardKernelGPU<float, int>(
            th_in_feat.template data<float>(), th_in_feat.size(0),
            th_out_feat.template data<float>(), out_nrows,
            th_num_nonzero.template data<float>(), th_in_feat.size(1),
            vec_in_map, vec_out_map, true, handle,
            at::cuda::getCurrentCUDAStream());
      } else if (th_in_feat.dtype() == torch::kFloat64) {
        NonzeroAvgPoolingForwardKernelGPU<float, int>(
            th_in_feat.template data<float>(), th_in_feat.size(0),
            th_out_feat.template data<float>(), out_nrows,
            th_num_nonzero.template data<float>(), th_in_feat.size(1),
            vec_in_map, vec_out_map, true, handle,
            at::cuda::getCurrentCUDAStream());
      } else {
        throw std::runtime_error("Dtype not supported.");
      }
#else
      throw std::runtime_error(
          "Minkowski Engine not compiled with GPU support. Please reinstall.");
#endif
    } else {
      auto vec_in_map = CopyToInOutMap<long>(th_in_map);
      auto vec_out_map = CopyToInOutMap<long>(th_out_map);
      if (th_in_feat.dtype() == torch::kFloat32) {
        NonzeroAvgPoolingForwardKernelCPU<float, long>(
            th_in_feat.template data<float>(),
            th_out_feat.template data<float>(),
            th_num_nonzero.template data<float>(), nchannel, vec_in_map,
            vec_out_map, out_nrows, true);
      } else if (th_in_feat.dtype() == torch::kFloat64) {
        NonzeroAvgPoolingForwardKernelCPU<double, long>(
            th_in_feat.template data<double>(),
            th_out_feat.template data<double>(),
            th_num_nonzero.template da>
  } else if (th_in_map.dtype() == torch::kInt32) {
    if (th_in_feat.is_cuda()) {
#ifndef CPU_ONLY
      auto vec_in_map = CopyToInOutMapGPU<int>(th_in_map);
      auto vec_out_map = CopyToInOutMapGPU<int>(th_out_map);

      if (th_in_feat.dtype() == torch::kFloat32) {
        NonzeroAvgPoolingForwardKernelGPU<float, int>(
            th_in_feat.template data<float>(), th_in_feat.size(0),
            th_out_feat.template data<float>(), out_nrows,
            th_num_nonzero.template data<float>(), th_in_feat.size(1),
            vec_in_map, vec_out_map, true, handle,
            at::cuda::getCurrentCUDAStream());
      } else if (th_in_feat.dtype() == torch::kFloat64) {
        NonzeroAvgPoolingForwardKernelGPU<float, int>(
            th_in_feat.template data<float>(), th_in_feat.size(0),
            th_out_feat.template data<float>(), out_nrows,
            th_num_nonzero.template data<float>(), th_in_feat.size(1),
            vec_in_map, vec_out_map, true, handle,
            at::cuda::getCurrentCUDAStream());
      } else {
        throw std::runtime_error("Dtype not supported.");
      }
#else
      throw std::runtime_error(
          "Minkowski Engine not compiled with GPU support. Please reinstall.");
#endif
    } else {
      auto vec_in_map = CopyToInOutMap<int>(th_in_map);
      auto vec_out_map = CopyToInOutMap<int>(th_out_map);
      if (th_in_feat.dtype() == torch::kFloat32) {
        NonzeroAvgPoolingForwardKernelCPU<float, int>(
            th_in_feat.template data<float>(),
            th_out_feat.template data<float>(),
            th_num_nonzero.template data<float>(), nchannel, vec_in_map,
            vec_out_map, out_nrows, true);
      } else if (th_in_feat.dtype() == torch::kFloat64) {
        NonzeroAvgPoolingForwardKernelCPU<double, int>(
            th_in_feat.template data<double>(),
            th_out_feat.template data<double>(),
            th_num_nonzero.template data<double>(), nchannel, vec_in_map,
            vec_out_map, out_nrows, true);
      } else {
        throw std::runtime_error("Dtype not supported.");
      }
    }
  }

  return th_out_feat;
}
*/

} // end namespace minkowski
