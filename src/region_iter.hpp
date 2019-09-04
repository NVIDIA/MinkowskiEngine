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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef REGION
#define REGION

#include "common.hpp"

template <typename Itype> class RegionIterator;
template <typename Itype> class Region {
public:
  Region(const Coord<Itype> &center_, const std::vector<int> &tensor_strides,
         const std::vector<int> &kernel_size, const std::vector<int> &dilations,
         int region_type, const Itype *p_offset, int n_offset);

  Region(const Coord<Itype> &lower_bound_,
         const std::vector<int> &tensor_strides,
         const std::vector<int> &kernel_size, const std::vector<int> &dilations,
         int region_type, const Itype *p_offset, int n_offset,
         bool use_lower_bound);

  RegionIterator<Itype> begin() { return RegionIterator<Itype>(*this); }
  RegionIterator<Itype> end() { return RegionIterator<Itype>(*this); }

  int D, region_type;
  std::vector<Itype> tensor_strides, kernel_size, dilations;
  const Itype *p_offset, n_offset;
  Coord<Itype> center;
  Coord<Itype> lb;
  Coord<Itype> ub;
  bool use_lower_bound;
};

template <typename Itype> class RegionIterator {
private:
  int D, curr_axis, offset_ind;
  const Region<Itype> &region;
  Coord<Itype> point;

public:
  bool done;
  RegionIterator(const Region<Itype> &region)
      : curr_axis(0), offset_ind(0), region(region), D(region.D), done(false) {
    // First point
    switch (region.region_type) {
    case 0:
      point = region.lb;
      break;
    case 1:
      // First, start from the origin
      point = region.center;
      break;
    case 2:
      point.resize(D + 1);
      // First offset
#ifdef BATCH_FIRST
      point[0] = region.center[0];
      for (int i = 1; i < D + 1; i++) {
        point[i] = region.center[i] + region.p_offset[i];
      }
#else
      for (int i = 0; i < D; i++) {
        point[i] = region.center[i] + region.p_offset[i];
      }
      point[D] = region.center[D];
#endif
      break;
    }
  }

  RegionIterator<Itype> &operator++() {
    switch (region.region_type) {
    case 0:
#ifdef BATCH_FIRST
      ASSERT(false, "Not implemented.");
#else
      // Iterate only from 0 to D-1, point[D] reserved for batch index
      for (int d = 0; d < D;) {
        point[d] += region.dilations[d] *
                    region.tensor_strides[d]; // point is initialized as lb
        if (point[d] <= region.ub[d])
          break;
        point[d] = region.lb[d];
        d++;
        if (d >= D) {
          done = true; // Signal to operator!= to end iteration
          break;
        }
      }
#endif
      return *this;
    case 1:
#ifdef BATCH_FIRST
      ASSERT(false, "Not implemented.");
#else
      while (curr_axis < D) {
        // Go through [4, 5, 1, 2] when kernel_size = 5, and ceter = 3.
        // Center passed at the initialization
        point[curr_axis] +=
            region.dilations[curr_axis] * region.tensor_strides[curr_axis];
        // skip if the current point is crossing the center
        if (point[curr_axis] == region.center[curr_axis]) {
          curr_axis++;
          continue;
        }
        // When it passes the last point, reset to the lower bound
        if (point[curr_axis] > region.ub[curr_axis])
          point[curr_axis] = region.lb[curr_axis];
        break;
      }
      if (curr_axis >= D) { // if it has past all axes
        done = true;
      }
#endif
      return *this;
    case 2:         // custom offset
      offset_ind++; // already past the first offset
#ifdef BATCH_FIRST
      ASSERT(false, "Not implemented.");
#else
      if (offset_ind >= region.n_offset) {
        done = true;
      } else {
        for (int i = 0; i < D; i++) {
          point[i] = region.center[i] + region.p_offset[D * offset_ind + i];
        }
      }
#endif
      return *this;
    }
    // To make the compiler happy
    return *this;
  }
  Coord<Itype> &operator*() { return point; }
};

// Only to be used for checking the end point of range based for loops.
template <typename Itype>
inline bool operator!=(const RegionIterator<Itype> &lhs,
                       const RegionIterator<Itype> &rhs) {
  return !lhs.done;
}

/**
 * Return the number of neighbors within the region.
 *
 * WARNING: must free *return_pairs after use.
 */
template <typename Itype>
std::vector<Itype> region_neighbors(
    const _CoordsHashMap<Itype> &in_coords_hashmap, const Coord<Itype> &coord,
    const std::vector<int> &tensor_strides, const std::vector<int> &kernel_size,
    const std::vector<int> &dilations, int region_type, const Itype *p_offset,
    int n_offset) {
  std::vector<Itype> pairs;
  auto region = Region<Itype>(coord, tensor_strides, kernel_size, dilations,
                              region_type, p_offset, n_offset);

  int kernel_ind = 0;
  for (auto &point : region) {
    auto in_coord_iter = in_coords_hashmap.find(point);
    if (in_coord_iter != in_coords_hashmap.end()) {
      pairs.push_back(kernel_ind);
      pairs.push_back(in_coord_iter->second);
      // in_map[kernel_ind][index] = in_coord_iter->second;
      // out_map[kernel_ind][index] = i;
    }
    kernel_ind++;
  }

  // Return memory and number of neighbors.
  return std::move(pairs);
}

#endif
