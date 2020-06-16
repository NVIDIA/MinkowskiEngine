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
#include "region.hpp"

namespace minkowski {

Region::Region(const vector<int> &tensor_strides_,
               const vector<int> &kernel_size_, const vector<int> &dilations_,
               int region_type_, const int *p_offset_, int n_offset_)
    : region_type(region_type_), tensor_strides(tensor_strides_),
      kernel_size(kernel_size_), dilations(dilations_), p_offset(p_offset_),
      n_offset(n_offset_) {
  D = tensor_strides.size();

  center.resize(D + 1);
  lb.resize(D + 1);
  ub.resize(D + 1);

  set_size();
}

Region::Region(const Region &region_)
    : region_type(region_.region_type), tensor_strides(region_.tensor_strides),
      kernel_size(region_.kernel_size), dilations(region_.dilations),
      p_offset(region_.p_offset), n_offset(region_.n_offset),
      size_(region_.size_) {
  D = tensor_strides.size();

  center.resize(D + 1);
  lb.resize(D + 1);
  ub.resize(D + 1);

  // set_size not required since it's copied from the other
}

Region::Region(const vector<int> &center_, const vector<int> &tensor_strides_,
               const vector<int> &kernel_size_, const vector<int> &dilations_,
               int region_type_, const int *p_offset_, int n_offset_)
    : region_type(region_type_), tensor_strides(tensor_strides_),
      kernel_size(kernel_size_), dilations(dilations_), p_offset(p_offset_),
      n_offset(n_offset_) {
  D = center_.size() - 1;

  center.resize(D + 1);
  lb.resize(D + 1);
  ub.resize(D + 1);

  set_size();
  set_bounds(center_);
}

void Region::set_bounds(const vector<int> &center_) {
  ASSERT(center_.size() == D + 1,
         "Size mismatch. input size: ", ArrToString(center_),
         ", D + 1 = ", D + 1);
  set_bounds(center_.data());
}

void Region::set_bounds(const int *p_center_) {
  std::copy_n(p_center_, D + 1, center.begin());

#ifdef BATCH_FIRST
  constexpr int batch_offset = 1;
  lb[0] = ub[0] = p_center_[0]; // set the batch index
#else
  constexpr int batch_offset = 0;
  lb[D] = ub[D] = p_center_[D]; // set the batch index
#endif

  for (int i = 0; i < D; i++) {
    // If the current kernel size is even, [0, 1, 2, 3] --> [0] for kernel
    // size 4.
    if (kernel_size[i] % 2 == 0) {
      lb[i + batch_offset] = p_center_[i + batch_offset];
      ub[i + batch_offset] =
          p_center_[i + batch_offset] +
          (kernel_size[i] - 1) * dilations[i] * tensor_strides[i];
    } else {
      lb[i + batch_offset] =
          p_center_[i + batch_offset] -
          int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
      ub[i + batch_offset] =
          p_center_[i + batch_offset] +
          int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    }
  }
}

RegionIterator::RegionIterator(const Region &region)
    : D(region.D), curr_axis(0), offset_ind(0), region(region), done(false) {
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
    constexpr int b = 1;
    point[0] = region.center[0];
#else
    constexpr int b = 0;
    point[D] = region.center[D];
#endif
    for (int i = 0; i < D; i++) {
      point[i + b] = region.center[i + b] + region.p_offset[i];
    }
    break;
  }
}

RegionIterator &RegionIterator::operator++() {
  switch (region.region_type) {
  case 0: {
#ifdef BATCH_FIRST
    constexpr int b = 1;
#else
    constexpr int b = 0;
#endif
    // Iterate only from 0 to D-1, point[D] reserved for batch index
    for (int d = 0; d < D;) {
      point[d + b] += region.dilations[d] *
                      region.tensor_strides[d]; // point is initialized as lb
      if (point[d + b] <= region.ub[d + b])
        break;
      point[d + b] = region.lb[d + b];
      d++;
      if (d >= D) {
        done = true; // Signal to operator!= to end iteration
        break;
      }
    }
    return *this;
  }
  case 1: {
#ifdef BATCH_FIRST
    constexpr int b = 1;
#else
    constexpr int b = 0;
#endif
    while (curr_axis < D) {
      // Go through [4, 5, 1, 2] when kernel_size = 5, and ceter = 3.
      // Center passed at the initialization
      point[curr_axis + b] +=
          region.dilations[curr_axis] * region.tensor_strides[curr_axis];
      // skip if the current point is crossing the center
      if (point[curr_axis + b] == region.center[curr_axis + b]) {
        curr_axis++;
        continue;
      }
      // When it passes the last point, reset to the lower bound
      if (point[curr_axis + b] > region.ub[curr_axis + b])
        point[curr_axis + b] = region.lb[curr_axis + b];
      break;
    }
    if (curr_axis >= D) { // if it has past all axes
      done = true;
    }
    return *this;
  }
  case 2: // custom offset
  {
#ifdef BATCH_FIRST
    constexpr int b = 1;
#else
    constexpr int b = 0;
#endif
    offset_ind++; // already past the first offset
    if (offset_ind >= region.n_offset) {
      done = true;
    } else {
      for (int i = 0; i < D; i++) {
        point[i + b] =
            region.center[i + b] + region.p_offset[D * offset_ind + i];
      }
    }
    return *this;
  }
  }
  // To make the compiler happy
  return *this;
}

} // end namespace minkowski
