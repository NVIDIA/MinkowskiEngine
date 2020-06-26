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
#include <cstdlib>
#include <vector>

#include "coordinate.hpp"
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include "gpu.cuh"
#endif

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
  kernel_region() = delete;
  MINK_CUDA_HOST_DEVICE kernel_region(
      REGION_TYPE::region_type type,
      size_type coordinate_size,      // Dimension of the coordinate
      size_type const *tensor_stride, // stride size between points
      size_type const *kernel_size,   // size of the kernel or region
      size_type const *dilation,      // stride / dilation within kernel,
      size_type const volume = 0,     // kernel volume
      coordinate_type const *p_offset = nullptr, // m_coordinate_size * n_offset
      uint32_t n_offset = 0)
      : m_region_type(type), m_coordinate_size{coordinate_size},
        m_num_offset{n_offset}, m_tensor_stride{tensor_stride},
        m_kernel_size{kernel_size},
        m_dilation{dilation}, m_volume{volume}, m_offset{p_offset} {
    if (m_volume == 0)
      set_volume();
  }

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
  MINK_CUDA_HOST_DEVICE inline size_type volume() const { return m_volume; }
  MINK_CUDA_HOST_DEVICE inline size_type coordinate_size() const {
    return m_coordinate_size;
  }
  MINK_CUDA_HOST_DEVICE inline size_type num_offset() const {
    return m_num_offset;
  }
  MINK_CUDA_HOST_DEVICE inline coordinate_type const *offset() const {
    return m_offset;
  }
  MINK_CUDA_HOST_DEVICE inline size_type const *tensor_stride() const {
    return m_tensor_stride;
  }
  MINK_CUDA_HOST_DEVICE inline size_type const *kernel_size() const {
    return m_kernel_size;
  }
  MINK_CUDA_HOST_DEVICE inline size_type const *dilation() const {
    return m_dilation;
  }

private:
  MINK_CUDA_HOST_DEVICE void set_volume() {
#ifndef __CUDA_ARCH__
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
#else
    // m_volume must be initialized when copied from cpu to gpu
#endif
  }

protected:
  // flag indicating tensor_stride, kernel_size, dilation are on gpu
  REGION_TYPE::region_type const m_region_type;
  size_type const m_coordinate_size;
  size_type m_num_offset, m_volume{0};

  size_type const *m_tensor_stride;
  size_type const *m_kernel_size;
  size_type const *m_dilation;

  coordinate_type const *m_offset;
  coordinate_type *m_lb;
  coordinate_type *m_ub;
  coordinate_type *m_tmp;
};

template <typename coordinate_type = default_types::dcoordinate_type>
class cpu_kernel_region : kernel_region<coordinate_type> {
public:
  using base_type = kernel_region<coordinate_type>;
  using self_type = cpu_kernel_region<coordinate_type>;
  using size_type = typename base_type::size_type;

public:
  cpu_kernel_region() = delete;
  cpu_kernel_region(
      REGION_TYPE::region_type type,
      size_type coordinate_size,      // Dimension of the coordinate
      size_type const *tensor_stride, // stride size between points
      size_type const *kernel_size,   // size of the kernel or region
      size_type const *dilation,      // stride / dilation within kernel,
      size_type const volume = 0,     // volume
      coordinate_type const *p_offset = nullptr, // m_coordinate_size * n_offset
      uint32_t n_offset = 0)
      : base_type{type,     coordinate_size, tensor_stride, kernel_size,
                  dilation, volume,          p_offset,      n_offset} {}

  using base_type::begin;
  using base_type::cbegin;
  using base_type::end;

  using base_type::coordinate_size;
  using base_type::num_offset;
  using base_type::offset;
  using base_type::region_type;
  using base_type::set_bounds;
  using base_type::volume;

#ifndef CPU_ONLY
  inline size_type const *device_tensor_stride() const {
    return m_d_tensor_stride;
  }
  inline size_type const *device_kernel_size() const { return m_d_kernel_size; }
  inline size_type const *device_dilation() const { return m_d_dilation; }
  inline coordinate_type const *device_offset() const { return m_d_offset; }

  self_type const to_gpu() {
    // move the kernel_region to GPU
    size_type num_bytes = (m_coordinate_size - 1) * 3 * sizeof(size_type);
    if (m_region_type == REGION_TYPE::CUSTOM)
      num_bytes +=
          (m_coordinate_size - 1) * m_num_offset * sizeof(coordinate_type);

    void *p_tmp = std::malloc(num_bytes);
    size_type *p_size_type = reinterpret_cast<size_type *>(p_tmp);
    coordinate_type *p_coordinate_type = reinterpret_cast<coordinate_type *>(
        p_size_type + 3 * (m_coordinate_size - 1));

    std::copy_n(m_tensor_stride, m_coordinate_size - 1, &p_size_type[0]);
    std::copy_n(m_kernel_size, m_coordinate_size - 1,
                &p_size_type[m_coordinate_size - 1]);
    std::copy_n(m_dilation, m_coordinate_size - 1,
                &p_size_type[2 * (m_coordinate_size - 1)]);

    if (m_region_type == REGION_TYPE::CUSTOM) {
      std::copy_n(m_offset, m_num_offset * (m_coordinate_size - 1),
                  p_coordinate_type);
    }

    LOG_DEBUG("Copied", num_bytes, "bytes to contiguous memory.");
    size_type *d_tmp;
    CUDA_CHECK(cudaMalloc((void **)&d_tmp, num_bytes));
    CUDA_CHECK(cudaMemcpy(d_tmp, p_tmp, num_bytes, cudaMemcpyHostToDevice));
    // clang-format off
    m_d_tensor_stride = d_tmp + 0 * (m_coordinate_size - 1);
    m_d_kernel_size   = d_tmp + 1 * (m_coordinate_size - 1);
    m_d_dilation      = d_tmp + 2 * (m_coordinate_size - 1);
    m_d_offset        = reinterpret_cast<coordinate_type*>(d_tmp + 3 * (m_coordinate_size - 1));
    // clang-format on

    m_on_gpu = true;

    std::free(p_tmp);

    return *this;
  }

  inline bool on_gpu() const { return m_on_gpu; }

  void clean() {
    if (m_on_gpu)
      CUDA_CHECK(cudaFree(m_d_tensor_stride));
  }
#endif

protected:
  using base_type::m_coordinate_size;
  using base_type::m_num_offset;
  using base_type::m_region_type;
  using base_type::m_volume;

  using base_type::m_dilation;
  using base_type::m_kernel_size;
  using base_type::m_tensor_stride;

  using base_type::m_lb;
  using base_type::m_offset;
  using base_type::m_tmp;
  using base_type::m_ub;

  bool m_on_gpu{false};

  // To move these to GPU, must move to gpu first
  size_type *m_d_tensor_stride;
  size_type *m_d_kernel_size;
  size_type *m_d_dilation;
  coordinate_type *m_d_offset;
};

/*
 * Kernel map that can be instantiated from CPU or purely on GPU.
 */
template <typename coordinate_type = default_types::dcoordinate_type>
class gpu_kernel_region : kernel_region<coordinate_type> {
public:
  using base_type = kernel_region<coordinate_type>;

public:
  // The input kernel_region should have initialized the m_d_tensor_stride ...
  gpu_kernel_region() = delete;
  MINK_CUDA_HOST_DEVICE
  gpu_kernel_region(cpu_kernel_region<coordinate_type> const &other)
      : base_type{other.region_type(),          other.coordinate_size(),
                  other.device_tensor_stride(), other.device_kernel_size(),
                  other.device_dilation(),      other.volume(),
                  other.device_offset(),        other.num_offset()} {}

  using base_type::begin;
  using base_type::cbegin;
  using base_type::end;

  using base_type::coordinate_size;
  using base_type::num_offset;
  using base_type::offset;
  using base_type::region_type;
  using base_type::set_bounds;
  using base_type::volume;

  using base_type::dilation;
  using base_type::kernel_size;
  using base_type::tensor_stride;

protected:
  using base_type::m_coordinate_size;
  using base_type::m_num_offset;
  using base_type::m_region_type;
  using base_type::m_volume;

  using base_type::m_dilation;
  using base_type::m_kernel_size;
  using base_type::m_tensor_stride;

  using base_type::m_lb;
  using base_type::m_offset;
  using base_type::m_tmp;
  using base_type::m_ub;
};

} // end namespace minkowski

#endif // REGION
