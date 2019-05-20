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
#include "region_iter.hpp"
#include "instantiation.hpp"

template <uint8_t D, typename Itype>
Region<D, Itype>::Region(const Coord<D, Itype> &center_,
                         const Arr<D, int> &tensor_strides,
                         const Arr<D, int> &kernel_size,
                         const Arr<D, int> &dilations, int region_type,
                         const Itype *p_offset, int n_offset)
    : region_type(region_type), tensor_strides(tensor_strides),
      kernel_size(kernel_size), dilations(dilations), p_offset(p_offset),
      n_offset(n_offset), use_lower_bound(false) {
  for (int i = 0; i < D; i++) {
    center[i] = center_[i];
    lb[i] =
        center_[i] - int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    ub[i] =
        center_[i] + int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
  }
  lb[D] = ub[D] = center[D] = center_[D]; // set the batch index
}

template <uint8_t D, typename Itype>
Region<D, Itype>::Region(const Coord<D, Itype> &lower_bound_,
                         const Arr<D, int> &tensor_strides,
                         const Arr<D, int> &kernel_size,
                         const Arr<D, int> &dilations, int region_type,
                         const Itype *p_offset, int n_offset,
                         bool use_lower_bound)
    : region_type(region_type), tensor_strides(tensor_strides),
      kernel_size(kernel_size), dilations(dilations), p_offset(p_offset),
      n_offset(n_offset), use_lower_bound(true) {
  if (region_type > 0)
    throw std::invalid_argument(
        Formatter() << "The region type " << region_type
                    << " is not supported with the use_lower_bound argument");
  for (int i = 0; i < D; i++) {
    lb[i] = lower_bound_[i];
    ub[i] = lower_bound_[i] + kernel_size[i] * dilations[i] * tensor_strides[i];
  }
  lb[D] = ub[D] = lower_bound_[D]; // set the batch index
}

INSTANTIATE_CLASS_DIM_ITYPE(Region, int32_t);
