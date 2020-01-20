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

#include <algorithm>
#include <vector>

#include "types.hpp"
#include "utils.hpp"

namespace minkowski {

class Region;
class RegionIterator {
private:
  int D, curr_axis, offset_ind;
  const Region &region;
  vector<int> point;

public:
  bool done;
  RegionIterator(const Region &region);
  RegionIterator &operator++();
  vector<int> operator*() { return point; };
};

class Region {
public:
  Region(const Region &region_);

  Region(const vector<int> &tensor_strides, const vector<int> &kernel_size,
         const vector<int> &dilations, int region_type, const int *p_offset,
         int n_offset);

  Region(const vector<int> &center_, const vector<int> &tensor_strides,
         const vector<int> &kernel_size, const vector<int> &dilations,
         int region_type, const int *p_offset, int n_offset);

  void set_bounds(const int *p_center_);
  void set_bounds(const vector<int> &center_);

  void set_size() {
    switch (region_type) {
    case 0:
      size_ = 1;
      for (const int curr_k : kernel_size)
        size_ *= curr_k;
      break;
    case 1:
      size_ = 1;
      ASSERT(std::all_of(kernel_size.begin(), kernel_size.end(),
                         [](int k) { return k > 2; }),
             "Invalid kernel size for hypercross: ", ArrToString(kernel_size));
      for (const int curr_k : kernel_size)
        size_ += (curr_k - 1);
      break;
    case 2:
      size_ = n_offset;
      break;
    };
  }

  int size() const { return size_; }

  RegionIterator begin() { return RegionIterator(*this); }
  RegionIterator end() { return RegionIterator(*this); }

  int D, region_type;
  const vector<int> &tensor_strides;
  const vector<int> &kernel_size;
  const vector<int> &dilations;
  const int *p_offset, n_offset;

  vector<int> center;
  vector<int> lb;
  vector<int> ub;
  int size_ = -1;
};

// Only to be used for checking the end point of range based for loops.
inline bool operator!=(const RegionIterator &lhs, const RegionIterator &rhs) {
  return !lhs.done;
}

} // end namespace minkowski

#endif // REGION
