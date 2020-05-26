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
#ifndef COORDS_MAP_CUH
#define COORDS_MAP_CUH

#include "types.hpp"
#include "types.cuh"

#include "3rdparty/hash/helper_functions.cuh"
#include "coords_functors.cuh"

#include <memory>
#include <thrust/device_vector.h>

namespace minkowski {

using cmap_types = ckey_value_types<int32_t, uint32_t>;
using ConcurrentCoordsToIndexMap = cmap_types::map_type;

template <typename MapType = ConcurrentCoordsToIndexMap> struct CoordsMap {

  void initialize_map(size_t coordinate_size, size_t estimated_size) {
    // Create coordinate map
    cmap = std::move(cmap_types::map_type::create(
        compute_hash_table_size(estimated_size),
        std::numeric_limits<cmap_types::ctype>::max(),
        coordinate<cmap_types::ctype>{nullptr},
        coordinate_murmur3<cmap_types::ctype>{coordinate_size},
        coordinate_equal_to<cmap_types::ctype>{coordinate_size}));
    CUDA_TRY(cudaStreamSynchronize(0));
  }

  // members
  std::unique_ptr<ctypes::map_type, std::function<void(ctypes::map_type *)>> cmap;
  thrust::device_vector<ctypes::coordinate_type> coordinates;
  std::size_t coordinate_size;
  std::size_t number_of_coordinates;

} // namespace minkowski

#endif // COORDS_MAP_CUH
