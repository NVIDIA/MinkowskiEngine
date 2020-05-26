/*
 *  Copyright 2020 NVIDIA Corporation.
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
 */

#include "coords_map.cuh"

#include <thrust/device_vector.h>

namespace minkowski {

template <>
CoordsMap<ConcurrentCoordsToIndexMap>::initialize(
    thrust::device_vector<coordinate_type> &&coordinates_,
    uint32_t number_of_coordinates_, uint32_t coordinate_size_,
    bool force_remap) {
  ASSERT(coordinates_.size() == number_of_coordinates_ * coordinate_size_,
         "Invalid sizes. coordinates.size(): ", coordinates_.size(),
         " != ", num_coordinates_, " * ", coordinate_size_);
  // update the member variables
  coordinate_size = coordinate_size_;
  coordinates = coordinates_; // transfer the ownership
  // TODO: Must overwrite after insertion, due to possible collision
  number_of_coordinates = number_of_coordinates_;
  // Allocate coordinate map
  initialize_map(coordinate_size, num_coordinates_);

  // Insert coordinates and get the result
  thrust::counting_iterator<uint32_t> coord_begin{0};
  thrust::device_vector<thrust::pair<bool, uint32_t>> insert_results{number_of_coordinates};
  thrust::transform(
      coord_begin, //
      coord_begin + num_coordinates_, //
      insert_results.begin(), //
      insert_coordinate<cmap_types::map_type, cmap_types::pair_type, ctypes::ctype>{
          *cmap, d_coords, coordinate_size});
  // Count the number of insertions
  CUDA_TRY(cudaStreamSynchronize(0));
  // TODO: update the number_of_coordinates to the valid number of coordinates
}

} // namespace minkowski
