/* Copyright (c) 2020 NVIDIA CORPORATION.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef STORAGE_CUH
#define STORAGE_CUH

#include "utils.hpp"

#include <vector>

namespace minkowski {

template <typename Dtype, typename ByteAllocator> class gpu_storage {
public:
  using data_type = Dtype;
  using byte_allocator_type = ByteAllocator;
  using self_type = gpu_storage<data_type, byte_allocator_type>;

  gpu_storage() : m_data(nullptr), m_num_elements(0) {}
  gpu_storage(uint64_t const num_elements) { allocate(num_elements); }
  gpu_storage(self_type const &other_storage) {
    LOG_DEBUG("copy storage constructor");
    if (other_storage.size() == 0)
      return;

    allocate(other_storage.size());
    CUDA_CHECK(cudaMemcpy(m_data, other_storage.cdata(),
                          other_storage.size() * sizeof(data_type),
                          cudaMemcpyDeviceToDevice));
  }
  gpu_storage(self_type &&other_storage) {
    LOG_DEBUG("move storage constructor from", other_storage.m_data,
              "with size", other_storage.m_num_elements);
    if (other_storage.size() == 0)
      return;
    m_num_elements = other_storage.size();
    m_data = other_storage.data();

    other_storage.m_data = nullptr;
    other_storage.m_num_elements = 0;
  }
  gpu_storage(std::vector<Dtype> const &vec) {
    LOG_DEBUG("vector storage constructor");
    from_vector(vec);
  }

  ~gpu_storage() { deallocate(); }

  data_type *allocate(uint64_t const num_elements) {
    if (num_elements == 0)
      return nullptr;
    m_num_elements = num_elements;
    m_data =
        (data_type *)m_allocator.allocate(m_num_elements * sizeof(data_type));
    LOG_DEBUG("Allocating", num_elements, "gpu storage at", m_data);
    return m_data;
  }

  void deallocate() {
    LOG_DEBUG("Deallocating", m_num_elements, "gpu storage at", m_data);
    if (m_num_elements > 0) {
      m_allocator.deallocate((char *)m_data,
                             m_num_elements * sizeof(data_type));
    }
  }

  void from_vector(std::vector<Dtype> const &vec) {
    resize(vec.size());
    if (m_num_elements > 0) {
      CUDA_CHECK(cudaMemcpy(m_data, vec.data(),
                            m_num_elements * sizeof(data_type),
                            cudaMemcpyHostToDevice));
    }
  }

  data_type *data() {
    check_pointer("data");
    return m_data;
  }
  data_type const *cdata() const {
    check_pointer("cdata");
    return m_data;
  }
  data_type *begin() {
    check_pointer("begin");
    return m_data;
  }
  data_type *end() {
    check_pointer("end");
    return m_data + m_num_elements;
  }
  data_type const *cbegin() const {
    check_pointer("cbegin");
    return m_data;
  }
  data_type const *cend() const {
    check_pointer("cend");
    return m_data + m_num_elements;
  }

  std::vector<data_type> to_vector() { return to_vector(size()); }
  std::vector<data_type> to_vector(uint64_t const num_elements) {
    std::vector<data_type> cpu_storage(num_elements);
    if (num_elements > 0)
      CUDA_CHECK(cudaMemcpy(cpu_storage.data(), m_data,
                            num_elements * sizeof(data_type),
                            cudaMemcpyDeviceToHost));
    return cpu_storage;
  }

  uint64_t size() const { return m_num_elements; }

  void resize(uint64_t const new_num_elements) {
    LOG_DEBUG("resizing from", m_num_elements, "to", new_num_elements);
    if (new_num_elements == m_num_elements)
      return;

    data_type *new_data =
        (data_type *)m_allocator.allocate(new_num_elements * sizeof(data_type));

    if (m_num_elements > 0) {
      CUDA_CHECK(cudaMemcpy(new_data, m_data,
                            new_num_elements * sizeof(data_type),
                            cudaMemcpyDeviceToDevice));
      m_allocator.deallocate((char *)m_data,
                             m_num_elements * sizeof(data_type));
    }
    m_data = new_data;
    m_num_elements = new_num_elements;
  }

  void print_by_vector(uint64_t const num_vec, uint64_t const vec_size) {
    auto const print_n = std::min(num_vec, size() / vec_size);
    auto const cpu_storage = to_vector(vec_size * print_n);
    for (int i = 0; i < print_n; ++i) {
      std::cout << PtrToString(&cpu_storage[i * vec_size], vec_size) << "\n";
    }
  }

private:
  void check_pointer(std::string const &fn) const {
#ifdef DEBUG
    if (m_data == nullptr) {
      throw std::runtime_error("storage.cuh: m_data == nullptr on " + fn);
    } else if (m_num_elements == 0) {
      throw std::runtime_error("storage.cuh: m_num_elements == 0 on" + fn);
    }
#endif
  }
  byte_allocator_type m_allocator;
  data_type *m_data = nullptr;
  uint64_t m_num_elements = 0;
};

template <typename Dtype, typename ByteAllocator>
void print(const gpu_storage<Dtype, ByteAllocator> &v) {
  auto cpu_storage = v.to_vector();
  for (size_t i = 0; i < cpu_storage.size(); i++)
    std::cout << " " << std::fixed << std::setprecision(3) << cpu_storage[i];
  std::cout << "\n";
}

// template void print(const thrust::device_vector<float> &v);
// template void print(const thrust::device_vector<int32_t> &v);

// template <typename Dtype1, typename Dtype2>
// void print(const thrust::device_vector<Dtype1> &v1,
//            const thrust::device_vector<Dtype2> &v2) {
//   for (size_t i = 0; i < v1.size(); i++)
//     std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
//   std::cout << "\n";
// }
//
// template void print(const thrust::device_vector<int32_t> &v1,
//                     const thrust::device_vector<int32_t> &v2);

} // namespace minkowski
#endif
