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

Region::Region(const vector<int> &tensor_strides_,
               const vector<int> &kernel_size_, const vector<int> &dilations_,
               int region_type_, const int *p_offset_, int n_offset_)
    : region_type(region_type_), tensor_strides(tensor_strides_),
      kernel_size(kernel_size_), dilations(dilations_), p_offset(p_offset_),
      n_offset(n_offset_), use_lower_bound(false) {
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
      use_lower_bound(false), size_(region_.size_) {
  D = tensor_strides.size();

  center.resize(D + 1);
  lb.resize(D + 1);
  ub.resize(D + 1);
}

Region::Region(const vector<int> &center_, const vector<int> &tensor_strides_,
               const vector<int> &kernel_size_, const vector<int> &dilations_,
               int region_type_, const int *p_offset_, int n_offset_)
    : region_type(region_type_), tensor_strides(tensor_strides_),
      kernel_size(kernel_size_), dilations(dilations_), p_offset(p_offset_),
      n_offset(n_offset_), use_lower_bound(false) {
  D = center_.size() - 1;

  center.resize(D + 1);
  lb.resize(D + 1);
  ub.resize(D + 1);

  set_size();
  set_bounds(center_);
}

Region::Region(const vector<int> &lower_bound_,
               const vector<int> &tensor_strides_,
               const vector<int> &kernel_size_, const vector<int> &dilations_,
               int region_type_, const int *p_offset_, int n_offset_,
               bool use_lower_bound_)
    : region_type(region_type_), tensor_strides(tensor_strides_),
      kernel_size(kernel_size_), dilations(dilations_), p_offset(p_offset_),
      n_offset(n_offset_), use_lower_bound(use_lower_bound_) {
  D = lower_bound_.size();

  center.resize(D + 1);

  set_size();
  set_bounds_with_lb(lower_bound_);
}

void Region::set_bounds_with_lb(const vector<int> &lb_) {
  lb = lb_; // input is the lower bound
  ub.resize(D + 1);

  if (region_type > 0)
    throw std::invalid_argument(
        Formatter() << "The region type " << region_type
                    << " is not supported with the use_lower_bound argument");
#ifdef BATCH_FIRST
  ub[0] = lb[0]; // set the batch index
  for (int i = 0; i < D; i++) {
    ub[i + 1] =
        lb[i + 1] + (kernel_size[i] - 1) * dilations[i] * tensor_strides[i];
  }
#else
  for (int i = 0; i < D; i++) {
    ub[i] = lb[i] + (kernel_size[i] - 1) * dilations[i] * tensor_strides[i];
  }
  ub[D] = lb[D]; // set the batch index
#endif
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
  lb[0] = ub[0] = p_center_[0]; // set the batch index
  for (int i = 0; i < D; i++) {
    lb[i + 1] = p_center_[i + 1] -
                int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    ub[i + 1] = p_center_[i + 1] +
                int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
  }
#else
  for (int i = 0; i < D; i++) {
    lb[i] = p_center_[i] -
            int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    ub[i] = p_center_[i] +
            int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
  }
  lb[D] = ub[D] = p_center_[D]; // set the batch index
#endif
}

void Region::set_bounds(const Coord<int> &center_) {
  center.resize(center_.size);
  std::copy_n(center_.ptr, center_.size, center.data());

#ifdef BATCH_FIRST
  lb[0] = ub[0] = center[0]; // set the batch index
  for (int i = 0; i < D; i++) {
    lb[i + 1] = center[i + 1] -
                int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    ub[i + 1] = center[i + 1] +
                int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
  }
#else
  for (int i = 0; i < D; i++) {
    lb[i] =
        center[i] - int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    ub[i] =
        center[i] + int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
  }
  lb[D] = ub[D] = center[D]; // set the batch index
#endif
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

RegionIterator &RegionIterator::operator++() {
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
