/*
 * Copyright (c) 2020 NVIDIA Corporation.
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
#ifndef KERNEL_MAP_HPP
#define KERNEL_MAP_HPP

#include "types.hpp"

#include <ostream>
#include <tuple>
#include <vector>

namespace minkowski {

/*
 * Kernel map specific types
 */
using cpu_in_map = default_types::index_vector_type;
using cpu_out_map = default_types::index_vector_type;

// Input index to output index mapping for each spatial kernel
using cpu_in_maps = std::vector<cpu_in_map>;
using cpu_out_maps = std::vector<cpu_out_map>;
struct cpu_kernel_map : std::pair<cpu_in_maps, cpu_out_maps> {
  using index_type = default_types::index_type;
  using index_pair =
      std::pair<default_types::index_type, default_types::index_type>;

  cpu_kernel_map() : std::pair<cpu_in_maps, cpu_out_maps>() {}
  cpu_kernel_map(std::pair<cpu_in_maps, cpu_out_maps> const &other)
      : std::pair<cpu_in_maps, cpu_out_maps>(other) {}

  // origin map initialization.
  cpu_kernel_map(std::vector<index_pair> &in_out,
                 std::vector<default_types::dcoordinate_type> const
                     &unique_batch_indicies) {
    auto comp = [](std::pair<index_type, index_type> const &l,
                   std::pair<index_type, index_type> const &r) {
      return l.second < r.second;
    };
    std::sort(in_out.begin(), in_out.end(), comp);

    auto const kernel_volume = unique_batch_indicies.size();
    this->first.resize(kernel_volume);
    this->second.resize(kernel_volume);

    for (index_type k = 0; k < unique_batch_indicies.size(); ++k) {
      auto const lb = std::lower_bound(in_out.begin(), in_out.end(),
                                       index_pair{0, k}, comp);
      auto const ub = std::upper_bound(in_out.begin(), in_out.end(),
                                       index_pair{0, k}, comp);
      auto const curr_size = ub - lb;
      default_types::index_type start_index = lb - in_out.begin();
      LOG_DEBUG("batch row_index:", k, "curr_size:", curr_size,
                "start_index:", start_index);
      // resize
      auto &in_map = this->first[k];
      auto &out_map = this->second[k];
      in_map.resize(curr_size);
      out_map.resize(curr_size);
      // copy
      for (uint32_t i = 0; i < curr_size; ++i) {
        auto const &curr_pair = in_out[i + start_index];
        in_map[i] = curr_pair.first;
        out_map[i] = curr_pair.second;
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  cpu_kernel_map const &kernel_map) {
    uint32_t map_size = 0;
    for (auto const &v : kernel_map.first) {
      map_size += v.size();
    }
    out << "cpu_kernel_map: number of unique maps:" << kernel_map.first.size()
        << ", kernel map size:" << map_size;
    return out;
  }
};

using cpu_kernel_map_reference = std::pair<cpu_in_maps &, cpu_out_maps &>;

} // namespace minkowski

#endif
