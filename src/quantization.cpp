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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "coordsmap.hpp"
#include "pooling_avg.hpp"
#ifndef CPU_ONLY
#include "pooling_avg.cuh"
#include <ATen/cuda/CUDAContext.h>
#endif
#include "utils.hpp"

namespace py = pybind11;

namespace minkowski {

struct IndexLabel {
  int index;
  int label;

  IndexLabel() : index(-1), label(-1) {}
  IndexLabel(int index_, int label_) : index(index_), label(label_) {}
};

using CoordsLabelMap =
    robin_hood::unordered_flat_map<vector<int>, IndexLabel, byte_hash_vec<int>>;

template <typename MapType>
vector<py::array> quantize_np(
    py::array_t<int, py::array::c_style | py::array::forcecast> coords) {
  py::buffer_info coords_info = coords.request();
  auto &shape = coords_info.shape;

  ASSERT(shape.size() == 2,
         "Dimension must be 2. The dimension of the input: ", shape.size());

  int *p_coords = (int *)coords_info.ptr;
  int nrows = shape[0], ncols = shape[1];

  // Create coords map
  CoordsMap<MapType> map;
  auto results = map.initialize_batch(p_coords, nrows, ncols, true, true);
  auto &mapping = std::get<0>(results);
  auto &inverse_mapping = std::get<1>(results);

  // Copy the concurrent vector to std vector
  py::array_t<int> py_mapping = py::array_t<int>(mapping.size());
  py::array_t<int> py_inverse_mapping =
      py::array_t<int>(inverse_mapping.size());

  py::buffer_info py_mapping_info = py_mapping.request();
  py::buffer_info py_inverse_mapping_info = py_inverse_mapping.request();
  int *p_py_mapping = (int *)py_mapping_info.ptr;
  int *p_py_inverse_mapping = (int *)py_inverse_mapping_info.ptr;

  std::copy_n(mapping.data(), mapping.size(), p_py_mapping);
  std::copy_n(inverse_mapping.data(), inverse_mapping.size(),
              p_py_inverse_mapping);

  // mapping is empty when coords are all unique
  return {py_mapping, py_inverse_mapping};
}

template <typename MapType> vector<at::Tensor> quantize_th(at::Tensor coords) {
  ASSERT(coords.dtype() == torch::kInt32,
         "Coordinates must be an int type tensor.");
  ASSERT(coords.dim() == 2,
         "Coordinates must be represnted as a matrix. Dimensions: ",
         coords.dim(), "!= 2.");

  CoordsMap<MapType> map;
  auto results = map.initialize_batch(
      coords.template data<int>(), coords.size(0), coords.size(1), true, true);
  auto mapping = std::get<0>(results);
  auto inverse_mapping = std::get<1>(results);

  // Long tensor for for easier indexing
  auto th_mapping = torch::empty({(long)mapping.size()},
                                 torch::TensorOptions().dtype(torch::kInt64));
  auto th_inverse_mapping =
      torch::empty({(long)inverse_mapping.size()},
                   torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_mapping = th_mapping.accessor<long int, 1>();
  auto a_th_inverse_mapping = th_inverse_mapping.accessor<long int, 1>();

  // Copy the output
  for (size_t i = 0; i < mapping.size(); ++i)
    a_th_mapping[i] = mapping[i];
  for (size_t i = 0; i < inverse_mapping.size(); ++i)
    a_th_inverse_mapping[i] = inverse_mapping[i];

  // mapping is empty when coords are all unique
  return {th_mapping, th_inverse_mapping};
}

vector<py::array> quantize_label_np(
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

  // Create coords map
  CoordsLabelMap map;
  map.reserve(nrows);
  for (int i = 0; i < nrows; i++) {
    vector<int> coord(ncols);
    std::copy_n(p_coords + i * ncols, ncols, coord.data());
    auto map_iter = map.find(coord);
    if (map_iter == map.end()) {
      map[move(coord)] = IndexLabel(i, p_labels[i]);
    } else if (map_iter->second.label != p_labels[i]) {
      map_iter->second.label = invalid_label;
    }
  }

  // Copy the concurrent vector to std vector
  py::array_t<int> py_mapping = py::array_t<int>(map.size());
  py::array_t<int> py_colabels = py::array_t<int>(map.size());

  py::buffer_info py_mapping_info = py_mapping.request();
  py::buffer_info py_colabels_info = py_colabels.request();
  int *p_py_mapping = (int *)py_mapping_info.ptr;
  int *p_py_colabels = (int *)py_colabels_info.ptr;

  int c = 0;
  for (const auto &kv : map) {
    p_py_mapping[c] = kv.second.index;
    p_py_colabels[c] = kv.second.label;
    c++;
  }

  return {py_mapping, py_colabels};
}

vector<at::Tensor> quantize_label_th(at::Tensor coords, at::Tensor labels,
                                     int invalid_label) {
  ASSERT(coords.dtype() == torch::kInt32,
         "Coordinates must be an int type tensor.");
  ASSERT(labels.dtype() == torch::kInt32, "Labels must be an int type tensor.");
  ASSERT(coords.dim() == 2,
         "Coordinates must be represnted as a matrix. Dimensions: ",
         coords.dim(), "!= 2.");
  ASSERT(coords.size(0) == labels.size(0),
         "Coords nrows must be equal to label size.");

  int *p_coords = coords.template data<int>();
  int *p_labels = labels.template data<int>();
  int nrows = coords.size(0), ncols = coords.size(1);

  // Create coords map
  CoordsLabelMap map;
  map.reserve(nrows);
  for (int i = 0; i < nrows; i++) {
    vector<int> coord(ncols);
    std::copy_n(p_coords + i * ncols, ncols, coord.data());
    auto map_iter = map.find(coord);
    if (map_iter == map.end()) {
      map[move(coord)] = IndexLabel(i, p_labels[i]);
    } else if (map_iter->second.label != p_labels[i]) {
      map_iter->second.label = invalid_label;
    }
  }

  // Copy the concurrent vector to std vector
  //
  // Long tensor for for easier indexing
  auto th_mapping = torch::empty({(long)map.size()},
                                 torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_mapping = th_mapping.accessor<long int, 1>();
  auto th_colabels = torch::empty({(long)map.size()},
                                  torch::TensorOptions().dtype(torch::kInt64));
  auto a_th_colabels = th_colabels.accessor<long int, 1>();

  int c = 0;
  for (const auto &kv : map) {
    a_th_mapping[c] = kv.second.index;
    a_th_colabels[c] = kv.second.label;
    c++;
  }

  return {th_mapping, th_colabels};
}

template vector<py::array> quantize_np<CoordsToIndexMap>(
    py::array_t<int, py::array::c_style | py::array::forcecast> coords);

template vector<at::Tensor> quantize_th<CoordsToIndexMap>(at::Tensor coords);

template <typename Dtype> InOutMaps<Dtype> CopyToInOutMap(at::Tensor th_map) {
  InOutMaps<Dtype> vec_map(1);
  vec_map[0].resize(th_map.size(0));
  std::copy_n(th_map.data<Dtype>(), th_map.size(0), vec_map[0].begin());
  return vec_map;
}

#ifndef CPU_ONLY
template <typename Dtype>
pInOutMaps<Dtype> CopyToInOutMapGPU(at::Tensor th_map) {
  pInOutMaps<Dtype> vec_map;

  Dtype *d_scr;
  CUDA_CHECK(cudaMalloc(&d_scr, th_map.size(0) * sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(d_scr, th_map.template data<Dtype>(),
                        th_map.size(0) * sizeof(Dtype),
                        cudaMemcpyHostToDevice));
  vec_map.push_back(pVector<Dtype>(d_scr, th_map.size(0)));
  return vec_map;
}
#endif

/**
 * A collection of feature averaging methods
 * mode == 0: non-weighted average
 * mode == 1: non-weighted sum
 * mode == k: TODO
 *
 * in_feat[in_map[i], j] --> out_feat[out_map[i], j]
 */
at::Tensor quantization_average_features(
    at::Tensor th_in_feat /* feature matrix */,
    at::Tensor th_in_map /* inverse_map from the quantization functions */,
    at::Tensor th_out_map /* range(N) */, int out_nrows,
    int mode /* average types */) {
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
            th_num_nonzero.template data<double>(), nchannel, vec_in_map,
            vec_out_map, out_nrows, true);
      } else {
        throw std::runtime_error("Dtype not supported.");
      }
    }
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

} // end namespace minkowski
