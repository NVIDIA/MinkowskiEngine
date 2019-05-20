/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do
 *  so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include <cstdint>

#include "utils.hpp"
#include "voxelization.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::vector<py::array_t<int>>
SparseVoxelization(py::array_t<uint64_t, py::array::c_style> keys,
                   py::array_t<int, py::array::c_style> labels,
                   int ignore_label, bool has_label) {
  py::buffer_info keys_info = keys.request(), labels_info = labels.request();
  if (keys_info.size != labels_info.size)
    throw std::invalid_argument(
        Formatter() << "Size of Keys and Labels mismatch. Size of keys: "
                    << std::to_string(keys_info.size) << ", size of labels: "
                    << std::to_string(labels_info.size));

  int num_points = keys_info.size;

  // Input pointers
  uint64_t *p_keys = (uint64_t *)keys_info.ptr;
  int *p_labels = (int *)labels_info.ptr;
  //  Return pointers
  int *p_return_key_inds, *p_return_labels;

  int final_n = sparse_voxelization(p_keys, p_labels, &p_return_key_inds,
                                    &p_return_labels, num_points, ignore_label,
                                    has_label);
  py::array_t<int> return_key_inds = py::array_t<int>(final_n);
  py::array_t<int> return_labels = py::array_t<int>(final_n);
  py::buffer_info return_key_inds_info = return_key_inds.request(),
                  return_labels_info = return_labels.request();
  int *p_py_return_key_inds = (int *)return_key_inds_info.ptr,
      *p_py_return_labels = (int *)return_labels_info.ptr;

  // Copy the temp outputs to the output numpy arrays
  memcpy(p_py_return_key_inds, p_return_key_inds, final_n * sizeof(int));
  memcpy(p_py_return_labels, p_return_labels, final_n * sizeof(int));
  free(p_return_key_inds);
  free(p_return_labels);

  // Return a pair
  std::vector<py::array_t<int>> return_vec;
  return_vec.push_back(return_key_inds);
  return_vec.push_back(return_labels);
  return return_vec;
}
