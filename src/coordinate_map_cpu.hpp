/* Copyright (c) 2020 NVIDIA CORPORATION.
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
#ifndef COORDINATE_MAP_CPU_HPP
#define COORDINATE_MAP_CPU_HPP

#include "coordinate_map.hpp"

namespace minkowski {

/*
 * Inherit from the CoordinateMap for a specific map type.
 */
template <typename coordinate_type>
class CoordinateMapCPU : public CoordinateMap<coordinate_type> {
public:
  // clang-format off
  using size_type   = typename CoordinateMap<coordinate_type>::size_type;
  using key_type    = typename CoordinateMap<coordinate_type>::key_type;
  using mapped_type = typename CoordinateMap<coordinate_type>::mapped_type;
  using value_type  = typename CoordinateMap<coordinate_type>::value_type;
  // clang-format on

public:
  CoordinateMapCPU(size_type const number_of_coordinates,
                   size_type const coordinate_size)
      : CoordinateMap<coordinate_type>(number_of_coordinates, coordinate_size) {
  }

  bool insert(key_type const &key, mapped_type const &val) {

    ASSERT(val < CoordinateMap<coordinate_type>::m_capacity,
           "Invalid mapped value: ", val,
           ", current capacity: ", CoordinateMap<coordinate_type>::m_capacity);
    coordinate_type *ptr =
        &CoordinateMap<coordinate_type>::m_coordinates
            [val * CoordinateMap<coordinate_type>::m_coordinate_size];
    std::copy_n(key.ptr, CoordinateMap<coordinate_type>::m_coordinate_size,
                ptr);
    auto insert_result = CoordinateMap<coordinate_type>::m_map.insert(
        value_type(coordinate<coordinate_type>{ptr}, val));
    if (insert_result.second) {
      return true;
    } else {
      return false;
    }
  }
};

} //end minkowski

#endif // COORDINATE_MAP_CPU
