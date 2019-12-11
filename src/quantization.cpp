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

#include "concurrent_coordsmap.hpp"
#include "utils.hpp"

namespace py = pybind11;

struct IndexLabel {
  int index;
  int label;

  IndexLabel() : index(-1), label(-1) {}
  IndexLabel(int index_, int label_) : index(index_), label(label_) {}
};

using ConcurrentCoordsLabelMap =
    tbb::concurrent_unordered_map<Coord<int>, IndexLabel, PointerCoordHash<int>,
                                  PointerEqualTo<int>>;

py::array
quantize(py::array_t<int, py::array::c_style | py::array::forcecast> coords) {
  py::buffer_info coords_info = coords.request();
  auto &shape = coords_info.shape;

  ASSERT(shape.size() == 2,
         "Dimension must be 2. The dimension of the input: ", shape.size());

  int *p_coords = (int *)coords_info.ptr;
  int nrows = shape[0], ncols = shape[1];

  // Create coords map
  ConcurrentCoordsMap map;

  // tbb::tick_count t0 = tbb::tick_count::now();
  //
  // Use the input order to define the map coords -> index
  tbb::parallel_for(tbb::blocked_range<int>(0, nrows),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord; // only a wrapper.
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        coord.ptr = &p_coords[ncols * i];
                        map[coord] = i;
                      }
                    },
                    tbb::auto_partitioner());

  // cout << "Creation: " << (t1 - t0).seconds() << endl;

  // If the mapping size is different, remap the entire coordinates
  if (nrows != map.size()) {
    // Assign a unique index to an item.
    //
    // Then assign the unique index to original row index mapping. Order does
    // not matter.  This randomized order (through multi-threads) will be the
    // new unique index.
    tbb::concurrent_vector<int> mapping;
    mapping.reserve(map.size());

    tbb::parallel_for(map.range(),
                      [&](decltype(map.map)::const_range_type &r) {
                        for (const auto &i : r) {
                          mapping.push_back(i.second);
                        }
                      },
                      tbb::auto_partitioner());

    // Copy the concurrent vector to std vector
    py::array_t<int> py_mapping = py::array_t<int>(mapping.size());
    py::buffer_info py_mapping_info = py_mapping.request();
    int *p_py_mapping = (int *)py_mapping_info.ptr;

    tbb::parallel_for(tbb::blocked_range<int>(0, mapping.size()),
                      [&](const tbb::blocked_range<int> &r) {
                        for (int i = r.begin(); i != r.end(); ++i) {
                          p_py_mapping[i] = mapping[i];
                        }
                      },
                      tbb::auto_partitioner());

    return py_mapping;
  } else {
    // Return null vector
    return py::array_t<int>(0);
  }
}

vector<py::array> quantize_label(
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
  ConcurrentCoordsLabelMap map;

  // tbb::tick_count t0 = tbb::tick_count::now();
  //
  // Use the input order to define the map coords -> index
  tbb::parallel_for(tbb::blocked_range<int>(0, nrows),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord; // only a wrapper.
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        coord.ptr = &p_coords[ncols * i];
                        map[coord] = IndexLabel(i, p_labels[i]);
                      }
                    },
                    tbb::auto_partitioner());

  // Set labels. All labels need to be set before to avoid race condition
  tbb::parallel_for(tbb::blocked_range<int>(0, nrows),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord; // only a wrapper.
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        coord.ptr = &p_coords[ncols * i];
                        auto &index_label = map[coord];
                        if (p_labels[i] != index_label.label) {
                          index_label.label = invalid_label;
                        }
                      }
                    },
                    tbb::auto_partitioner());

  // Unlike the function `quantize`, we need to return the mappping and
  // corresponding labels as the label order is arbitrary.

  // cout << "Creation: " << (t1 - t0).seconds() << endl;

  // Assign a unique index to an item.
  //
  // Then assign the unique index to original row index mapping. Order does
  // not matter.  This randomized order (through multi-threads) will be the
  // new unique index.
  tbb::concurrent_vector<IndexLabel> map_labels;
  map_labels.reserve(map.size());

  // Copy the map_labels and corresponding label
  tbb::parallel_for(map.range(),
                    [&](decltype(map)::const_range_type &r) {
                      for (const auto &i : r) {
                        map_labels.push_back(i.second);
                      }
                    },
                    tbb::auto_partitioner());

  // Copy the concurrent vector to std vector
  py::array_t<int> py_mapping = py::array_t<int>(map.size());
  py::array_t<int> py_colabels = py::array_t<int>(map.size());

  py::buffer_info py_mapping_info = py_mapping.request();
  py::buffer_info py_colabels_info = py_colabels.request();
  int *p_py_mapping = (int *)py_mapping_info.ptr;
  int *p_py_colabels = (int *)py_colabels_info.ptr;

  tbb::parallel_for(tbb::blocked_range<int>(0, map.size()),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        const auto &index_label = map_labels[i];
                        p_py_mapping[i] = index_label.index;
                        p_py_colabels[i] = index_label.label;
                      }
                    },
                    tbb::auto_partitioner());

  vector<py::array> return_pair = {py_mapping, py_colabels};
  return return_pair;
}
