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
#include "kernel_map.hpp"
#include "kernel_region.hpp"
#include <numeric>
#include <omp.h>
#include <torch/extension.h>

namespace minkowski {

namespace detail {

template <typename coordinate_type,
          typename stride_type = default_types::stride_type>
bool is_coordinate_aligned(coordinate<coordinate_type> const &point,
                           stride_type const &stride) {
  for (uint32_t i = 0; i < stride.size(); ++i) {
    if (point[i + 1] % stride[i] != 0)
      return false;
  }
  return true;
}

template <typename coordinate_type, typename Dtype, typename MapType>
std::pair<at::Tensor, at::Tensor>
field_map_kernel(uint32_t const num_tfield,      //
                 uint32_t const coordinate_size, //
                 Dtype const *const p_tfield,    //
                 MapType const &in_map,          //
                 default_types::stride_type const &tensor_stride) {
  constexpr bool is_float32 = std::is_same<Dtype, float>::value;
  at::ScalarType const float_type =
      is_float32 ? at::ScalarType::Float : at::ScalarType::Double;

  cpu_in_maps in_maps = initialize_maps<cpu_in_map>(1, num_tfield);
  cpu_out_maps out_maps = initialize_maps<cpu_out_map>(1, num_tfield);

  uint32_t num_used{0};

  // OMP
  // size_t stride = max((size_t)100, numElements / (2 *
  // omp_get_max_threads())); size_t N = (numElements + stride - 1) /
  // stride;

  // compute the chunk size per thread.
  // There's a trade-off between the thread initialization overhead and
  // the job sizes. If some jobs finish earlier than others due to
  // imbalance in hash distribution, these threads will be idle.
  const size_t N = 2 * omp_get_max_threads();
  const size_t stride = (num_tfield + N - 1) / N;
  LOG_DEBUG("kernel map with", N, "chunks and", stride, "stride.");

#pragma omp parallel for
  for (uint32_t n = 0; n < N; n++) {
    // temporary variables for each thread
    std::vector<coordinate_type> curr_vec(coordinate_size);
    coordinate<coordinate_type> curr_coordinate(curr_vec.data());
    uint32_t curr_index_begin;

    for (auto i = stride * n;
         i < std::min<uint64_t>((n + 1) * stride, uint64_t(num_tfield)); ++i) {

      // batch index
      curr_vec[0] = std::lroundf(p_tfield[i * coordinate_size]);
      for (uint32_t j = 1; j < coordinate_size; ++j) {
        auto const curr_tensor_stride = tensor_stride[j - 1];
        curr_vec[j] =
            curr_tensor_stride *
            std::floor(p_tfield[coordinate_size * i + j] / curr_tensor_stride);
      }

      const auto iter_in = in_map.find(curr_coordinate);
      // LOG_DEBUG(kernel_ind, ":",
      //           PtrToString(iter_out->first.data(),
      //           coordinate_size),
      //           "->", PtrToString(point.data(),
      //           coordinate_size));
      if (iter_in != in_map.end()) {
#pragma omp atomic capture
        {
          curr_index_begin = num_used;
          num_used += 1;
        }
        // Ensure that in_maps and out_maps are resized accordingly
        in_maps[0][curr_index_begin] = iter_in->second;
        out_maps[0][curr_index_begin] = i;
        // LOG_DEBUG(kernel_ind, ":",
        //           PtrToString(iter_in->first.data(),
        //           coordinate_size),
        //           "->",
        //           PtrToString(iter_out->first.data(),
        //           coordinate_size));
      }
    }
  }

  auto final_in_map = torch::empty(
      {num_used},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
  auto final_out_map = torch::empty(
      {num_used},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));

  std::copy_n(in_maps[0].data(), num_used,
              final_in_map.template data_ptr<int>());
  std::copy_n(out_maps[0].data(), num_used,
              final_out_map.template data_ptr<int>());

  return std::make_pair(final_in_map, final_out_map);
}

template <typename coordinate_type, typename Dtype, typename MapType>
std::vector<at::Tensor> interpolation_map_weight_kernel(
    uint32_t const num_tfield,      //
    uint32_t const coordinate_size, //
    Dtype const *const p_tfield,    //
    MapType const &in_map,          //
    default_types::stride_type const &tensor_stride) {
  constexpr bool is_float32 = std::is_same<Dtype, float>::value;
  at::ScalarType const float_type =
      is_float32 ? at::ScalarType::Float : at::ScalarType::Double;
  uint32_t const neighbor_volume = std::pow(2, (coordinate_size - 1));
  LOG_DEBUG("neighbor_volume :", neighbor_volume, "num_tfield:", num_tfield);

  cpu_in_maps in_maps =
      initialize_maps<cpu_in_map>(neighbor_volume, num_tfield);
  cpu_out_maps out_maps =
      initialize_maps<cpu_out_map>(neighbor_volume, num_tfield);

  std::vector<std::vector<Dtype>> weights =
      initialize_maps<std::vector<Dtype>>(neighbor_volume, num_tfield);

  std::vector<uint32_t> num_used(neighbor_volume);
  std::for_each(num_used.begin(), num_used.end(), [](auto &i) { i = 0; });

  // OMP
  // size_t stride = max((size_t)100, numElements / (2 *
  // omp_get_max_threads())); size_t N = (numElements + stride - 1) /
  // stride;

  // compute the chunk size per thread.
  // There's a trade-off between the thread initialization overhead and
  // the job sizes. If some jobs finish earlier than others due to
  // imbalance in hash distribution, these threads will be idle.
  const size_t N = 2 * omp_get_max_threads();
  const size_t stride = (num_tfield + N - 1) / N;
  LOG_DEBUG("kernel map with", N, "chunks and", stride, "stride.");

#pragma omp parallel for
  for (uint32_t n = 0; n < N; n++) {
    // temporary variables for each thread
    std::vector<coordinate_type> curr_vec(coordinate_size), lb(coordinate_size),
        ub(coordinate_size);
    coordinate<coordinate_type> curr_coordinate(curr_vec.data());
    uint32_t curr_index_begin;

    for (auto i = stride * n;
         i < std::min<uint64_t>((n + 1) * stride, uint64_t(num_tfield)); ++i) {

      // batch index
      curr_vec[0] = std::lroundf(p_tfield[i * coordinate_size]);
      for (uint32_t j = 1; j < coordinate_size; ++j) {
        auto const curr_tensor_stride = tensor_stride[j - 1];
        lb[j] =
            curr_tensor_stride *
            std::floor(p_tfield[coordinate_size * i + j] / curr_tensor_stride);
        ub[j] = lb[j] + curr_tensor_stride;
        curr_vec[j] = lb[j];
      }

      // For elements in the current region
      for (uint32_t neighbor_ind = 0; neighbor_ind < neighbor_volume;
           ++neighbor_ind) {

        uint32_t mask = 1;
        for (uint32_t j = coordinate_size - 1; j > 0; --j) {
          if ((neighbor_ind & mask) == 0)
            curr_vec[j] = lb[j];
          else
            curr_vec[j] = ub[j];
          mask = mask << 1;
        }

        const auto iter_in = in_map.find(curr_coordinate);
        // LOG_DEBUG(kernel_ind, ":",
        //           PtrToString(iter_out->first.data(),
        //           coordinate_size),
        //           "->", PtrToString(point.data(),
        //           coordinate_size));
        if (iter_in != in_map.end()) {
#pragma omp atomic capture
          {
            curr_index_begin = num_used[neighbor_ind];
            num_used[neighbor_ind] += 1;
          }
          // Compute weights
          Dtype weight = 1.0;
          for (uint32_t j = 1; j < coordinate_size; ++j) {
            weight *=
                1 - std::abs(p_tfield[coordinate_size * i + j] - curr_vec[j]) /
                        tensor_stride[j - 1];
          }

          // Ensure that in_maps and out_maps are resized accordingly
          in_maps[neighbor_ind][curr_index_begin] = iter_in->second;
          out_maps[neighbor_ind][curr_index_begin] = i;
          weights[neighbor_ind][curr_index_begin] = weight;
          // LOG_DEBUG(kernel_ind, ":",
          //           PtrToString(iter_in->first.data(),
          //           coordinate_size),
          //           "->",
          //           PtrToString(iter_out->first.data(),
          //           coordinate_size));
        }
      }
    }
  }

  auto const total_num_used =
      std::accumulate(num_used.begin(), num_used.end(), 0);

  auto final_in_map = torch::empty(
      {total_num_used},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
  auto final_out_map = torch::empty(
      {total_num_used},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
  auto final_weights = torch::empty(
      {total_num_used},
      torch::TensorOptions().dtype(float_type).requires_grad(false));

  uint32_t final_begin = 0;
  for (uint32_t i = 0; i < neighbor_volume; ++i) {
    uint32_t const max_num = num_used[i];
    LOG_DEBUG("kernel index", i, "size:", max_num);

    std::copy_n(in_maps[i].data(), max_num,
                &final_in_map.template data_ptr<int>()[final_begin]);
    std::copy_n(out_maps[i].data(), max_num,
                &final_out_map.template data_ptr<int>()[final_begin]);
    std::copy_n(weights[i].data(), max_num,
                final_weights.template data_ptr<Dtype>() + final_begin);

    final_begin += max_num;
  }
  return {final_in_map, final_out_map, final_weights};
}

} // namespace detail

/*
 * Inherit from the CoordinateMap for a specific map type.
 */
// clang-format off
template <typename coordinate_type,
          template <typename T> class TemplatedAllocator = std::allocator>
class CoordinateMapCPU : public CoordinateMap<coordinate_type, TemplatedAllocator> {
public:
  using base_type                 = CoordinateMap<coordinate_type, TemplatedAllocator>;
  using self_type                 = CoordinateMapCPU<coordinate_type, TemplatedAllocator>;
  using size_type                 = typename base_type::size_type;
  using index_type                = typename base_type::index_type;
  using stride_type               = typename base_type::stride_type;

  using key_type       = coordinate<coordinate_type>;
  using mapped_type    = default_types::index_type;
  using hasher         = detail::coordinate_murmur3<coordinate_type>;
  using key_equal      = detail::coordinate_equal_to<coordinate_type>;
  using map_type       =
      robin_hood::unordered_flat_map<key_type,    // key
                                     mapped_type, // mapped_type
                                     hasher,      // hasher
                                     key_equal    // equality
                                     >;

  using value_type                = typename map_type::value_type;
  using iterator                  = typename map_type::iterator;
  using const_iterator            = typename map_type::const_iterator;

  using index_vector_type         = typename base_type::index_vector_type;
  using byte_allocator_type       = TemplatedAllocator<char>;
  // clang-format on

public:
  CoordinateMapCPU() = delete;
  CoordinateMapCPU(size_type const number_of_coordinates,
                   size_type const coordinate_size,
                   stride_type const &stride = {1},
                   byte_allocator_type alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, alloc),
        m_map(
            map_type{0, hasher{coordinate_size}, key_equal{coordinate_size}}) {
    m_map.reserve(number_of_coordinates);
  }

  /*
   * @brief given a key iterator begin-end pair and a value iterator begin-end
   * pair, insert all elements.
   *
   * @return none
   */
  void insert(coordinate_type const *coordinate_begin,
              coordinate_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;
    base_type::allocate(N);
    index_type value = 0;
    for (coordinate_type const *key = coordinate_begin; key != coordinate_end;
         key += m_coordinate_size, ++value) {
      // value_type ctor needed because this might be called with std::pair's
      insert(key_type(key), value);
    }
  }

  /*
   * @brief given a key iterator begin-end pair and a value iterator begin-end
   * pair, insert all elements.
   *
   * @return pair<vector<long>, vector<long>> if return_unique_inverse_map.
   * mapping is a vector of unique indices and inverse_mapping is a vector of
   * indices that reconstructs the original coordinate from the list of unique
   * coordinates.
   *
   * >>> unique_coordinates = input_coordinates[mapping]
   * >>> reconstructed_coordinates = unique_coordinates[inverse_mapping]
   * >>> torch.all(reconstructed_coordinates == input_coordinates)
   */
  template <bool remap>
  std::pair<std::vector<int64_t>, std::vector<int64_t>> // return maps
  insert_and_map(coordinate_type const *coordinate_begin,
                 coordinate_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;

    std::vector<int64_t> mapping, inverse_mapping;
    base_type::allocate(N);
    mapping.reserve(N);
    inverse_mapping.reserve(N);

    index_type value{0}, row_index{0};
    for (coordinate_type const *key = coordinate_begin; key != coordinate_end;
         key += m_coordinate_size, row_index += 1) {
      // value_type ctor needed because this might be called with std::pair's
      auto const result = insert(key_type(key), value);
      if (result.second) {
        mapping.push_back(row_index);
        inverse_mapping.push_back(value);
      } else {
        // result.first is an iterator of pair<key, mapped_type>
        inverse_mapping.push_back(result.first->second);
      }
      value += remap ? result.second : 1;
    }

    return std::make_pair(std::move(mapping), std::move(inverse_mapping));
  }

  /*
   * @brief given a key iterator begin-end pair find all valid keys and its
   * index.
   *
   * @return a pair of (valid index, query value) vectors.
   */
  template <typename key_iterator>
  std::pair<index_vector_type, index_vector_type> find(key_iterator key_first,
                                                       key_iterator key_last) {
    size_type N = key_last - key_first;
    ASSERT(N <= base_type::m_capacity,
           "Invalid search range. Current capacity:", base_type::m_capacity,
           ", search range:", N);

    // reserve the result slots
    index_vector_type valid_query_index, query_result;
    valid_query_index.reserve(N);
    query_result.reserve(N);

    key_iterator key_curr{key_first};
    for (; key_curr != key_last; ++key_curr) {
      auto const query_iter = m_map.find(*key_curr);
      // If valid query
      if (query_iter != m_map.end()) {
        valid_query_index.push_back(key_curr - key_first);
        query_result.push_back(query_iter->second);
      }
    }
    return std::make_pair(valid_query_index, query_result);
  }

  // Network specific functions.

  /*
   * @brief strided coordinate map.
   */
  self_type stride(stride_type const &stride) const {
    ASSERT(stride.size() == m_coordinate_size - 1, "Invalid stride", stride);
    // Over estimate the reserve size to be size();
    self_type stride_map(
        size(), m_coordinate_size,
        detail::stride_tensor_stride(base_type::m_tensor_stride, stride),
        base_type::m_byte_allocator);

    index_type c = 0;
    std::vector<coordinate_type> dst(m_coordinate_size);
    coordinate<coordinate_type> strided_coordinate(&dst[0]);
    for (auto const &kv : m_map) {
      detail::stride_coordinate<coordinate_type>(kv.first, dst,
                                                 stride_map.m_tensor_stride);
      auto result = stride_map.insert(strided_coordinate, c);
      c += result.second;
    }

    return stride_map;
  }

  /*****************************************************************************
   * Map generation
   ****************************************************************************/

  /*
   * @brief strided coordinate map for region.
   */
  self_type stride_region(cpu_kernel_region<coordinate_type> const &kernel,
                          stride_type const &out_tensor_stride) const {
    ASSERT(kernel.coordinate_size() == m_coordinate_size, "Invalid kernel");
    // Over estimate the reserve size to be size();
    self_type stride_map(size() * kernel.volume(), m_coordinate_size,
                         out_tensor_stride, base_type::m_byte_allocator);

    auto ckernel = cpu_kernel_region<coordinate_type>(kernel);
    std::vector<coordinate_type> tmp(m_coordinate_size);
    coordinate<coordinate_type> point(tmp.data());

    index_type num_used{0};
    if (kernel.is_transpose()) {
      for (auto iter_in = m_map.begin(); iter_in != m_map.end(); ++iter_in) {

        // For elements in the current region
        for (uint32_t kernel_ind = 0; kernel_ind < ckernel.volume();
             ++kernel_ind) {
          ckernel.coordinate_at(kernel_ind, iter_in->first.data(), tmp.data());
          auto const result = stride_map.insert(point, num_used);
          num_used += result.second;
        }
      }
    } else {
      LOG_DEBUG("stride_region with no transpose");
      // Expand coordinates with regular conv
      for (auto iter_in = m_map.begin(); iter_in != m_map.end(); ++iter_in) {
        // For elements in the current region
        for (uint32_t kernel_ind = 0; kernel_ind < ckernel.volume();
             ++kernel_ind) {
          // TODO replace with more efficient code
          ckernel.coordinate_at(kernel_ind, iter_in->first.data(), tmp.data());
          if (detail::is_coordinate_aligned<coordinate_type, stride_type>(
                  point, out_tensor_stride)) {
            auto const result = stride_map.insert(point, num_used);
            num_used += result.second;
          }
        }
      }
    }
    return stride_map;
  }

  /*
   * @brief strided coordinate map.
   */
  self_type origin() const {
    // tensor stride is set to {0,..., 0} for the origin map.
    stride_type origin_tensor_stride(m_coordinate_size - 1);
    std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                  [](auto &i) { i = 0; });

    // Over estimate the reserve size to be size();
    self_type origin_map(size(), m_coordinate_size, origin_tensor_stride,
                         base_type::m_byte_allocator);

    index_type c = 0;
    std::vector<coordinate_type> dst(m_coordinate_size);
    std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });

    coordinate<coordinate_type> tmp_coordinate(&dst[0]);
    for (auto const &kv : m_map) {
      dst[0] = kv.first[0];
      auto result = origin_map.insert(tmp_coordinate, c);
      c += result.second;
    }
    return origin_map;
  }

  /*
   * @brief generate a new coordinate map that only keeps coordinates with true
   * keep mask
   */
  self_type prune(bool const *keep_begin, bool const *keep_end) const {
    ASSERT(keep_end - keep_begin == size(), "Invalid range for pruning");

    // Over estimate the reserve size to be size();
    self_type pruned_map(size(), m_coordinate_size, base_type::m_tensor_stride,
                         base_type::m_byte_allocator);

    index_type c = 0;
    for (auto const &kv : m_map) {
      // Use the row index defined
      if (keep_begin[kv.second]) {
        auto result = pruned_map.insert(kv.first, c);
        c += result.second;
      }
    }
    LOG_DEBUG("size:", pruned_map.size(), "capacity:", pruned_map.capacity());
    return pruned_map;
  }

  self_type merge(const self_type &other) const {
    std::vector<std::reference_wrapper<self_type>> maps{*this, other};
    // maps.push_back(*this);
    // maps.push_back(other);
    return merge(maps);
  }

  self_type
  merge(const std::vector<std::reference_wrapper<self_type>> &maps) const {
    // merge all input maps
    size_t all_size = std::accumulate(
        maps.begin(), maps.end(), 0,
        [](size_t sum, const self_type &map) { return sum + map.size(); });
    self_type merged_map(all_size, m_coordinate_size,
                         base_type::m_tensor_stride,
                         base_type::m_byte_allocator);
    // Push all coordinates
    index_type c = 0;
    for (self_type const &map : maps) {
      for (auto const &kv : map.m_map) {
        auto result = merged_map.insert(kv.first, c);
        c += result.second;
      }
    }

    return merged_map;
  }

  /*****************************************************************************
   * Kernel map
   ****************************************************************************/
  cpu_kernel_map
  kernel_map(self_type const &out_coordinate_map,
             cpu_kernel_region<coordinate_type> const &kernel) const {
    // Over estimate the reserve size to be size();
    size_type out_size = out_coordinate_map.size();
    size_type kernel_volume = kernel.volume();
    LOG_DEBUG("kernel volume:", kernel_volume, "out_size:", out_size);

    cpu_in_maps in_maps = initialize_maps<cpu_in_map>(kernel_volume, out_size);
    cpu_out_maps out_maps =
        initialize_maps<cpu_out_map>(kernel_volume, out_size);
    std::vector<size_type> num_used(kernel_volume);
    std::for_each(num_used.begin(), num_used.end(), [](auto &i) { i = 0; });

    // OMP
    const auto &out_mmap = out_coordinate_map.m_map;
    const size_t out_map_num_elements = out_mmap.capacity();

    // size_t stride = max((size_t)100, numElements / (2 *
    // omp_get_max_threads())); size_t N = (numElements + stride - 1) /
    // stride;

    // compute the chunk size per thread.
    // There's a trade-off between the thread initialization overhead and the
    // job sizes. If some jobs finish earlier than others due to imbalance in
    // hash distribution, these threads will be idle.
    size_t N = 2 * omp_get_max_threads();
    const size_t stride = (out_map_num_elements + N - 1) / N;
    N = (out_map_num_elements + stride - 1) / stride;
    LOG_DEBUG("kernel map with", N, "chunks and", stride, "stride.");
    LOG_DEBUG((kernel.region_type() != RegionType::CUSTOM && kernel_volume == 1)
                  ? "single kernel"
                  : "otherwise");

    // When no need to iterate through the region
    // Put if outside the loop for speed
    if (kernel.region_type() != RegionType::CUSTOM && kernel_volume == 1) {
      index_type curr_index_begin = 0;
      for (auto iter_out = out_mmap.begin(); iter_out != out_mmap.end();
           ++iter_out) {
        const auto iter_in = m_map.find(iter_out->first);
        if (iter_in != m_map.end()) {
          in_maps[0][curr_index_begin] = iter_in->second;
          out_maps[0][curr_index_begin] = iter_out->second;
          ++curr_index_begin;
        }
      }
      num_used[0] = curr_index_begin;
    } else {
#pragma omp parallel for
      for (index_type n = 0; n < N; n++) {
        auto ckernel = cpu_kernel_region<coordinate_type>(kernel);
        // temporary variables for each thread
        std::vector<coordinate_type> tmp(m_coordinate_size);
        coordinate<coordinate_type> curr_kernel_coordinate(tmp.data());

        index_type curr_index_begin;
        for (auto iter_out = out_mmap.begin(stride * n);
             iter_out.num_steps() <
             std::min(stride, out_map_num_elements - n * stride);
             ++iter_out) {

          // For elements in the current region
          for (uint32_t kernel_ind = 0; kernel_ind < ckernel.volume();
               ++kernel_ind) {
            // If the input coord exists
            ckernel.coordinate_at(kernel_ind, iter_out->first.data(),
                                  tmp.data());
            const auto iter_in = m_map.find(curr_kernel_coordinate);
            // LOG_DEBUG(kernel_ind, ":",
            //           PtrToString(iter_out->first.data(), m_coordinate_size),
            //           "->", PtrToString(point.data(), m_coordinate_size));
            if (iter_in != m_map.end()) {
#pragma omp atomic capture
              {
                curr_index_begin = num_used[kernel_ind];
                num_used[kernel_ind] += 1;
              }
              // Ensure that in_maps and out_maps are resized accordingly
              in_maps[kernel_ind][curr_index_begin] = iter_in->second;
              out_maps[kernel_ind][curr_index_begin] = iter_out->second;
              // LOG_DEBUG(kernel_ind, ":",
              //           PtrToString(iter_in->first.data(),
              //           m_coordinate_size),
              //           "->",
              //           PtrToString(iter_out->first.data(),
              //           m_coordinate_size));
            }
          }
        }
      }
    }

    for (index_type i = 0; i < kernel_volume; ++i) {
      index_type max_num = num_used[i];
      LOG_DEBUG("kernel index", i, "size:", max_num);
      in_maps[i].resize(max_num);
      out_maps[i].resize(max_num);
    }

    return std::make_pair(in_maps, out_maps);
  }

  cpu_kernel_map stride_map(self_type const &out_coordinate_map,
                            stride_type const &out_tensor_stride) const {
    // generate an in-out (kernel) map that maps all input points in the same
    // voxel to strided output voxel.
    size_type in_size = size();
    LOG_DEBUG("Generate stride_map with in NNZ:", in_size,
              "out NNZ:", out_coordinate_map.size(),
              "out_tensor_stride:", out_tensor_stride);
    cpu_in_maps in_maps = initialize_maps<cpu_in_map>(1, in_size);
    cpu_out_maps out_maps = initialize_maps<cpu_out_map>(1, in_size);

    LOG_DEBUG("stride map in_maps.size():", in_size);
    LOG_DEBUG("stride map out_maps.size():", out_coordinate_map.size());
    // compute the chunk size per thread.
    // There's a trade-off between the thread initialization overhead and the
    // job sizes. If some jobs finish earlier than others due to imbalance in
    // hash distribution, these threads will be idle.
    const size_t in_map_num_elements = m_map.capacity();
    size_t N = 2 * omp_get_max_threads();
    const size_t stride = (in_map_num_elements + N - 1) / N;
    N = (in_map_num_elements + stride - 1) / stride;
    LOG_DEBUG("kernel map with", N, "chunks.");

    index_type num_used = 0;
#pragma omp parallel for
    for (index_type n = 0; n < N; ++n) {
      index_type curr_index_begin;
      std::vector<coordinate_type> dst(m_coordinate_size);
      for (auto iter_in = m_map.begin(stride * n);
           iter_in.num_steps() <
           std::min(stride, in_map_num_elements - n * stride);
           ++iter_in) {
        detail::stride_coordinate<coordinate_type>(iter_in->first, dst,
                                                   out_tensor_stride);
        const auto iter_out =
            out_coordinate_map.find(coordinate<coordinate_type>(dst.data()));
        ASSERT(iter_out != out_coordinate_map.m_map.cend(),
               "Invalid out_coordinate_map");
#pragma omp atomic capture
        {
          curr_index_begin = num_used;
          num_used += 1;
        }

        in_maps[0][curr_index_begin] = iter_in->second;
        out_maps[0][curr_index_begin] = iter_out->second;
      }
    }

    return std::make_pair(move(in_maps), move(out_maps));
  }

  cpu_kernel_map origin_map(self_type const &origin_coordinate_map) const {
    // generate an in-out (kernel) map that maps all input points in the same
    // voxel to strided output voxel.
    ASSERT(std::all_of(origin_coordinate_map.get_tensor_stride().begin(),
                       origin_coordinate_map.get_tensor_stride().end(),
                       [](auto const &i) { return i == 0; }),
           "Invalid origin tensor stride",
           origin_coordinate_map.get_tensor_stride());

    size_type const in_size = size();
    size_type const out_size = origin_coordinate_map.size();
    LOG_DEBUG("Generate origin_map with in NNZ:", in_size,
              "out NNZ:", out_size);
    ASSERT(in_size > out_size, "Invalid out_coordinate_map");

    std::vector<std::pair<index_type, index_type>> in_out(in_size);

    // compute the chunk size per thread.
    // There's a trade-off between the thread initialization overhead and the
    // job sizes. If some jobs finish earlier than others due to imbalance in
    // hash distribution, these threads will be idle.
    size_t const in_map_num_elements = m_map.capacity();
    size_t N = 2 * omp_get_max_threads();
    size_t const stride = (in_map_num_elements + N - 1) / N;
    N = (in_map_num_elements + stride - 1) / stride;
    LOG_DEBUG("kernel map with", N, "chunks.");

    size_type num_used = 0;
#pragma omp parallel for
    for (index_type n = 0; n < N; ++n) {
      index_type curr_index_begin;
      std::vector<coordinate_type> dst(m_coordinate_size);
      std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });

      for (auto iter_in = m_map.begin(stride * n);
           iter_in.num_steps() <
           std::min(stride, in_map_num_elements - n * stride);
           ++iter_in) {
        dst[0] = iter_in->first[0];
        const auto iter_origin =
            origin_coordinate_map.find(coordinate<coordinate_type>(dst.data()));
        ASSERT(iter_origin != origin_coordinate_map.m_map.cend(),
               "Invalid origin_coordinate_map");
        index_type origin_row_index = iter_origin->second;

#pragma omp atomic capture
        {
          curr_index_begin = num_used;
          num_used += 1;
        }

        in_out[curr_index_begin] =
            std::make_pair(iter_in->second, origin_row_index);
      }
    }

    // Decomposed kernel map
    auto batch_indices = origin_coordinate_map.batch_indices();
    return cpu_kernel_map(in_out, batch_indices);
  }

  /*****************************************************************************
   * Interpolation
   ****************************************************************************/

  /*
   * Given a continuous tensor field, return the weights and associated kernel
   * map
   */
  std::vector<at::Tensor>
  interpolation_map_weight(at::Tensor const &tfield) const {
    // Over estimate the reserve size to be size();
    ASSERT(tfield.dim() == 2, "Invalid tfield dimension");
    ASSERT(tfield.size(1) == m_coordinate_size, "Invalid tfield size");

    // AT_DISPATCH_FLOATING_TYPES(
    //     tfield.scalar_type(), "interpolation_map_weight_kernel", [&] {
    switch (tfield.scalar_type()) {
    case at::ScalarType::Double:
      return detail::interpolation_map_weight_kernel<coordinate_type, double,
                                                     map_type>(
          tfield.size(0),                     //
          m_coordinate_size,                  //
          tfield.template data_ptr<double>(), //
          m_map,                              //
          base_type::m_tensor_stride);
    case at::ScalarType::Float:
      return detail::interpolation_map_weight_kernel<coordinate_type, float,
                                                     map_type>(
          tfield.size(0),                    //
          m_coordinate_size,                 //
          tfield.template data_ptr<float>(), //
          m_map,                             //
          base_type::m_tensor_stride);
    default:
      ASSERT(false, "Unsupported float type");
    }
  }

  template <typename coordinate_field_type>
  std::pair<at::Tensor, at::Tensor>
  field_map(coordinate_field_type const *p_tfield,
            size_type const num_tfield) const {
    return detail::field_map_kernel<coordinate_type, coordinate_field_type,
                                    map_type>(num_tfield,        //
                                              m_coordinate_size, //
                                              p_tfield,          //
                                              m_map,             //
                                              base_type::m_tensor_stride);
  }

  /*****************************************************************************
   * Union Map
   ****************************************************************************/

  /*
   * Find mapping from all inputs to self. The mapping is a list of 2xN tensors
   */
  std::vector<at::Tensor> union_map(
      std::vector<std::reference_wrapper<self_type>> const &in_maps) const {
    std::vector<at::Tensor> in_out_maps;

    for (self_type const &in_map : in_maps) {
      size_type const N = in_map.size();
      size_type const capacity = in_map.m_map.capacity();
      auto curr_map = torch::empty(
          {2, N},
          torch::TensorOptions().dtype(torch::kInt64).requires_grad(false));
      int64_t *p_in_rows = curr_map.template data_ptr<int64_t>();
      std::fill_n(p_in_rows, 2 * N, -1);
      int64_t *p_union_rows = p_in_rows + N;

      index_type offset{0};
      for (auto iter_in = in_map.m_map.begin(); iter_in.num_steps() < capacity;
           ++iter_in) {
        p_in_rows[offset] = iter_in->second;
        // WARNING: This assumes that the union map has a corresponding
        // coordinate for speed. This will results in an unexpected behavior
        // if the union map does not contain the coordinate.
        p_union_rows[offset] = find(iter_in->first)->second;
        LOG_DEBUG("offset:", offset, " in:", p_in_rows[offset],
                  " out:", p_union_rows[offset]);
        offset++;
      }

      in_out_maps.push_back(std::move(curr_map));
    }

    return in_out_maps;
  }

  inline size_type size() const noexcept { return m_map.size(); }
  std::string to_string() const {
    Formatter o;
    o << "CoordinateMapCPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }

  using base_type::capacity;
  using base_type::coordinate_size;
  using base_type::get_tensor_stride;

  inline void reserve(size_type c) {
    base_type::reserve(c);
    m_map.reserve(c);
  }

  void copy_coordinates(coordinate_type *dst_coordinate) const {
    if (m_map.size() == 0)
      return;
    size_t const capacity = m_map.capacity();
    size_t N = omp_get_max_threads();
    const size_t stride = (capacity + N - 1) / N;
    N = (capacity + stride - 1) / stride;
    LOG_DEBUG("kernel map with", N, "chunks, stride", stride, "capacity",
              capacity);

    // When no need to iterate through the region
    // Put if outside the loop for speed
#pragma omp parallel for
    for (index_type n = 0; n < N; ++n) {
      for (auto it = m_map.begin(stride * n);                        //
           it.num_steps() < std::min(stride, capacity - n * stride); //
           ++it) {
        std::copy_n(it->first.data(), m_coordinate_size,
                    dst_coordinate + m_coordinate_size * it->second);
      }
    }
  }

  std::vector<coordinate_type> batch_indices() const {
    std::vector<coordinate_type> indices(size());
    for (auto it = m_map.begin(); it != m_map.end(); ++it) {
      indices[it->second] = it->first.data()[0];
    }
    return indices;
  }

  std::pair<iterator, bool> insert(key_type const &key,
                                   mapped_type const &val) {
    ASSERT(val < base_type::m_capacity, "Invalid mapped value: ", val,
           ", current capacity: ", base_type::m_capacity);
    coordinate_type *ptr = &base_type::m_coordinates[val * m_coordinate_size];
    std::copy_n(key.data(), m_coordinate_size, ptr);
    return m_map.insert(value_type(coordinate<coordinate_type>{ptr}, val));
  }

  inline iterator find(key_type const &key) { return m_map.find(key); }
  inline const_iterator find(key_type const &key) const {
    return m_map.find(key);
  }
  inline const_iterator cend() const { return m_map.cend(); }

private:
  using base_type::m_coordinate_size;
  map_type m_map;
};

// Field map
template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator = std::allocator>
class CoordinateFieldMapCPU
    : public CoordinateMap<coordinate_field_type, TemplatedAllocator> {
  // Coordinate wrapper
public:
  using base_type = CoordinateMap<coordinate_field_type, TemplatedAllocator>;
  using coordinate_map_type =
      CoordinateMapCPU<coordinate_int_type, TemplatedAllocator>;
  using self_type =
      CoordinateFieldMapCPU<coordinate_field_type, coordinate_int_type,
                            TemplatedAllocator>;
  using size_type = typename base_type::size_type;
  using index_type = typename base_type::index_type;
  using stride_type = typename base_type::stride_type;
  using byte_allocator_type = TemplatedAllocator<char>;

public:
  CoordinateFieldMapCPU() = delete;
  CoordinateFieldMapCPU(size_type const number_of_coordinates,
                        size_type const coordinate_size,
                        stride_type const &stride = {1},
                        byte_allocator_type alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, alloc),
        m_size(number_of_coordinates) {
    base_type::reserve(number_of_coordinates);
  }

  /*
   * @brief given a key iterator begin-end pair and a value iterator begin-end
   * pair, insert all elements.
   *
   * @return none
   */
  void insert(coordinate_field_type const *coordinate_begin,
              coordinate_field_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;
    base_type::allocate(N);
    // copy data directly to the ptr
    std::copy_n(coordinate_begin, N * m_coordinate_size,
                base_type::coordinate_data());
  }

  using base_type::const_coordinate_data;
  using base_type::coordinate_data;

  void copy_coordinates(coordinate_field_type *dst_coordinate) const {
    std::copy_n(base_type::const_coordinate_data(), size() * m_coordinate_size,
                dst_coordinate);
  }

  void quantize_coordinates(coordinate_int_type *p_dst_coordinates,
                            stride_type const &tensor_stride) const {
    coordinate_field_type const *const p_tfield = const_coordinate_data();
    int64_t const stride_prod = std::accumulate(
        tensor_stride.begin(), tensor_stride.end(), 1, std::multiplies<>());
    ASSERT(stride_prod > 0, "Invalid stride");

    const size_t N = omp_get_max_threads();
    const size_t stride = (size() + N - 1) / N;
    LOG_DEBUG("kernel map with", N, "chunks and", stride, "stride.");

    if (stride_prod == 1) {
#pragma omp parallel for
      for (uint32_t n = 0; n < N; n++) {
        for (auto i = stride * n;
             i < std::min<uint64_t>((n + 1) * stride, uint64_t(size())); ++i) {

          // batch index
          coordinate_int_type *p_curr_dst =
              &p_dst_coordinates[i * m_coordinate_size];
          p_curr_dst[0] = std::lroundf(p_tfield[i * m_coordinate_size]);
          for (uint32_t j = 1; j < m_coordinate_size; ++j) {
            p_curr_dst[j] = std::floor(p_tfield[m_coordinate_size * i + j]);
          }
        }
      }
    } else {
#pragma omp parallel for
      for (uint32_t n = 0; n < N; n++) {
        for (auto i = stride * n;
             i < std::min<uint64_t>((n + 1) * stride, uint64_t(size())); ++i) {

          // batch index
          coordinate_int_type *p_curr_dst =
              &p_dst_coordinates[i * m_coordinate_size];
          p_curr_dst[0] = std::lroundf(p_tfield[i * m_coordinate_size]);
          for (uint32_t j = 1; j < m_coordinate_size; ++j) {
            auto const curr_tensor_stride = tensor_stride[j - 1];
            p_curr_dst[j] = curr_tensor_stride *
                            std::floor(p_tfield[m_coordinate_size * i + j] /
                                       curr_tensor_stride);
          }
        }
      }
    }
  }

  coordinate_map_type origin() const {
    // tensor stride is set to {0,..., 0} for the origin map.
    stride_type origin_tensor_stride(m_coordinate_size - 1);
    std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                  [](auto &i) { i = 0; });

    // Over estimate the reserve size to be size();
    CoordinateMapCPU<coordinate_int_type, TemplatedAllocator> origin_map(
        size(), m_coordinate_size, origin_tensor_stride,
        base_type::m_byte_allocator);

    coordinate_field_type const *const p_tfield = const_coordinate_data();

    std::vector<coordinate_int_type> dst(m_coordinate_size);
    std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });
    coordinate<coordinate_int_type> tmp_coordinate(&dst[0]);

    index_type c = 0;
    for (size_t i = 0; i < size(); ++i) {
      dst[0] = coordinate_int_type(p_tfield[i * m_coordinate_size]);
      auto result = origin_map.insert(tmp_coordinate, c);
      c += result.second;
    }
    return origin_map;
  }

  cpu_kernel_map
  origin_map(coordinate_map_type const &origin_coordinate_map) const {
    // generate an in-out (kernel) map that maps all input points in the same
    // voxel to strided output voxel.
    ASSERT(std::all_of(origin_coordinate_map.get_tensor_stride().begin(),
                       origin_coordinate_map.get_tensor_stride().end(),
                       [](auto const &i) { return i == 0; }),
           "Invalid origin tensor stride",
           origin_coordinate_map.get_tensor_stride());

    size_type const in_size = size();
    size_type const out_size = origin_coordinate_map.size();
    LOG_DEBUG("Generate origin_map with in NNZ:", in_size,
              "out NNZ:", out_size);
    ASSERT(in_size > out_size, "Invalid out_coordinate_map");

    std::vector<std::pair<index_type, index_type>> in_out(in_size);

    // compute the chunk size per thread.
    // There's a trade-off between the thread initialization overhead and the
    // job sizes. If some jobs finish earlier than others due to imbalance in
    // hash distribution, these threads will be idle.
    size_t const in_map_num_elements = size();
    size_t N = 2 * omp_get_max_threads();
    size_t const stride = (in_map_num_elements + N - 1) / N;
    N = (in_map_num_elements + stride - 1) / stride;
    LOG_DEBUG("kernel map with", N, "chunks", stride, "stride");
    coordinate_field_type const *const p_tfield = const_coordinate_data();
    size_type num_used = 0;
#pragma omp parallel for
    for (index_type n = 0; n < N; ++n) {
      index_type curr_index_begin;
      std::vector<coordinate_int_type> dst(m_coordinate_size);
      std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });
      // LOG_DEBUG(n, "th chunk contains",
      //           std::min((n + 1) * stride, in_map_num_elements) - n * stride,
      //           "elements");

      for (size_t i = stride * n;
           i < std::min((n + 1) * stride, in_map_num_elements); ++i) {
        dst[0] = p_tfield[i * m_coordinate_size];
        // LOG_DEBUG(i, "th row batch index", dst[0]);
        const auto iter_origin = origin_coordinate_map.find(
            coordinate<coordinate_int_type>(dst.data()));
        ASSERT(iter_origin != origin_coordinate_map.cend(),
               "Invalid origin_coordinate_map");
        index_type origin_row_index = iter_origin->second;

#pragma omp atomic capture
        {
          curr_index_begin = num_used;
          num_used += 1;
        }

        in_out[curr_index_begin] = std::make_pair(i, origin_row_index);
      }
    }

    // #ifdef DEBUG
    //     LOG_DEBUG("Kernel map");
    //     for (auto iter : in_out) {
    //       std::cout << iter.first << ":" << iter.second << "\n";
    //     }
    // #endif
    // Decomposed kernel map
    auto batch_indices = origin_coordinate_map.batch_indices();
    return cpu_kernel_map(in_out, batch_indices);
  }

  inline size_type size() const noexcept { return m_size; }
  std::string to_string() const {
    Formatter o;
    o << "CoordinateFieldMapCPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }

private:
  using base_type::m_coordinate_size;
  size_type m_size;
};

} // namespace minkowski

#endif // COORDINATE_MAP_CPU
