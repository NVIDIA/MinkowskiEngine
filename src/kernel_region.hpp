/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu)
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

#include "coordinate.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace minkowski {

namespace REGION_TYPE {

enum region_type { HYPER_CUBE, HYPER_CROSS, CUSTOM };

}

// A wrapper for a convolution kernel or a pooling kernel.
template <typename coordinate_type = default_types::dcoordinate_type>
class kernel_region {
public:
  using index_type = default_types::index_type;

public:
  class kernel_region_iterator {
  public:
    // clang-format off
    using self_type         = kernel_region_iterator;

    // iterator traits
    using iterator_category = std::forward_iterator_tag;
    using value_type        = coordinate<coordinate_type>;
    using difference_type   = int32_t;
    using pointer           = coordinate<coordinate_type>*;
    using reference         = coordinate<coordinate_type>&;
    // clang-format on

  public:
    kernel_region_iterator() = delete;
    // Take the temporary memory space `p_coordinate` for dereference.
    MINK_CUDA_HOST_DEVICE
    kernel_region_iterator(coordinate_type *tmp, kernel_region const &region)
        : done(false), m_region{region}, m_tmp{tmp}, m_coordinate(m_tmp) {
      if (m_tmp != nullptr) {
        for (index_type i = 0; i < m_region.m_coordinate_size; ++i) {
          m_tmp[i] = m_region.m_lb[i];
        }
        // LOG_DEBUG("KernelRegionIterator tmp:",
        //           PtrToString(m_tmp, m_region.m_coordinate_size));
      }
    }

    // reference operator*();
    // pointer   operator->();
    MINK_CUDA_HOST_DEVICE inline reference operator*() noexcept {
      return m_coordinate;
    }
    MINK_CUDA_HOST_DEVICE inline pointer operator->() noexcept {
      return &m_coordinate;
    }

    /*
     * Cannot use the coordinates for hash-map insertion with * or ->.
     */

    // this_type& operator++();
    // this_type  operator++(int);
    MINK_CUDA_HOST_DEVICE inline self_type operator++() noexcept {
      // Iterate only from 1 to m_coordinate_size, 0th element is reserved for
      // batch index.
      for (index_type m_axis = 0;;) {
        m_tmp[m_axis + 1] +=
            m_region.m_dilation[m_axis] * m_region.m_tensor_stride[m_axis];
        if (m_tmp[m_axis + 1] <= m_region.m_ub[m_axis + 1]) {
          break;
        }
        m_tmp[m_axis + 1] = m_region.m_lb[m_axis + 1];
        ++m_axis;
        if (m_axis >= m_region.m_coordinate_size - 1) {
          done = true; // Signal to operator!= to end iteration
          break;
        }
      }
      return *this;
    }
    // MINK_CUDA_HOST_DEVICE inline self_type operator++(int) noexcept;
    // TDOO: % based iteration

    MINK_CUDA_HOST_DEVICE inline bool
    operator!=(self_type const &other) const noexcept {
      return !done;
    }

  private:
    bool done;
    kernel_region const &m_region;
    coordinate_type *m_tmp;
    coordinate<coordinate_type> m_coordinate;
  };

public:
  // clang-format off
  using size_type      = default_types::size_type;
  using stride_type    = default_types::stride_type;
  using iterator       = kernel_region_iterator;
  using const_iterator = const kernel_region_iterator;
  // clang-format on

public:
  MINK_CUDA_HOST_DEVICE kernel_region(
      REGION_TYPE::region_type type,
      size_type coordinate_size,      // Dimension of the coordinate
      size_type const *tensor_stride, // stride size between points
      size_type const *kernel_size,   // size of the kernel or region
      size_type const *dilation,      // stride / dilation within kernel,
      coordinate_type const *p_offset = nullptr, // m_coordinate_size * n_offset
      uint32_t n_offset = 0)
      : m_region_type(type), m_coordinate_size{coordinate_size},
        m_num_offset{n_offset}, m_tensor_stride{tensor_stride},
        m_kernel_size{kernel_size}, m_dilation{dilation}, m_offset{p_offset} {
    set_volume();
    // set the memory space
  }

  MINK_CUDA_HOST_DEVICE inline size_type volume() const { return m_volume; }

  /*
   * initialize memory and set the bounds
   */
  MINK_CUDA_HOST_DEVICE void
  set_bounds(coordinate_type const *p_center,
             coordinate_type *p_lb, // lower bound temporary memory space.
                                    // Management should be done outside.
             coordinate_type *p_ub, coordinate_type *p_tmp) {
    m_tmp = p_tmp;
    m_lb = p_lb;
    m_ub = p_ub;
    m_lb[0] = p_center[0]; // set the batch index
    constexpr index_type batch_offset = 1;

    for (index_type i = 0; i < m_coordinate_size - 1; ++i) {
      // If the current kernel size is even, [0, 1, 2, 3] --> [0] for kernel
      // size 4.
      if (m_kernel_size[i] % 2 == 0) {
        m_lb[i + batch_offset] = p_center[i + batch_offset];
        m_ub[i + batch_offset] =
            p_center[i + batch_offset] +
            (m_kernel_size[i] - 1) * m_dilation[i] * m_tensor_stride[i];
      } else {
        m_lb[i + batch_offset] =
            p_center[i + batch_offset] -
            int(m_kernel_size[i] / 2) * m_dilation[i] * m_tensor_stride[i];
        m_ub[i + batch_offset] =
            p_center[i + batch_offset] +
            int(m_kernel_size[i] / 2) * m_dilation[i] * m_tensor_stride[i];
      }
    }
    // LOG_DEBUG("KernelRegion lower bound:",
    //           PtrToString(m_lb, m_coordinate_size));
    // LOG_DEBUG("KernelRegion upper bound:",
    //           PtrToString(m_ub, m_coordinate_size));
  }

  MINK_CUDA_HOST_DEVICE iterator begin() { return iterator(m_tmp, *this); }
  MINK_CUDA_HOST_DEVICE const_iterator cbegin() const {
    return const_iterator(m_tmp, *this);
  }
  MINK_CUDA_HOST_DEVICE iterator end() { return iterator(nullptr, *this); }
  MINK_CUDA_HOST_DEVICE kernel_region_iterator end() const {
    return const_iterator(nullptr, *this);
  }

  MINK_CUDA_HOST_DEVICE REGION_TYPE::region_type region_type() const {
    return m_region_type;
  }

private:
  MINK_CUDA_HOST_DEVICE void set_volume() {
    switch (m_region_type) {
    case REGION_TYPE::HYPER_CUBE:
      m_volume = 1;
      for (index_type i = 0; i < m_coordinate_size - 1; ++i)
        m_volume *= m_kernel_size[i];
      break;
    case REGION_TYPE::HYPER_CROSS:
      m_volume = 1;
      for (index_type i = 0; i < m_coordinate_size - 1; ++i)
        m_volume += (m_kernel_size[i] - 1);
      break;
    case REGION_TYPE::CUSTOM:
      m_volume = m_num_offset;
      break;
    };
  }

private:
  REGION_TYPE::region_type const m_region_type;
  size_type const m_coordinate_size;
  size_type m_num_offset, m_volume;
  // all needs to be loaded on the shared memory for GPU.
  size_type const *m_tensor_stride;
  size_type const *m_kernel_size;
  size_type const *m_dilation;

  coordinate_type const *m_offset;
  coordinate_type *m_lb;
  coordinate_type *m_ub;
  coordinate_type *m_tmp;
};

// Only to be used for checking the end point of range based for loops.
// inline bool operator!=(const RegionIterator &lhs, const RegionIterator &rhs)
// {
//   return !lhs.done;
// }

} // end namespace minkowski

#endif // REGION
