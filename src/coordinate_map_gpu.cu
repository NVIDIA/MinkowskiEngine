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
#include "coordinate_map_functors.cuh"
#include "coordinate_map_gpu.cuh"
#include "gpu.cuh"
#include "kernel_map.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace minkowski {

/*
 * @brief Given a key iterator begin-end pair and a value iterator begin-end
 * pair, insert all elements.
 *
 * @note The key and value iterators can be 1) pointers, 2) coordinate or vector
 * iterators.
 *
 * @return none
 */
template <typename coordinate_type, typename MapAllocator,
          typename CoordinateAllocator, typename KernelMapAllocator>
template <typename mapped_iterator>
void CoordinateMapGPU<
    coordinate_type, MapAllocator, CoordinateAllocator,
    KernelMapAllocator>::insert(coordinate_iterator<coordinate_type> key_first,
                                coordinate_iterator<coordinate_type> key_last,
                                mapped_iterator value_first,
                                mapped_iterator value_last) {
  size_type const N = key_last - key_first;
  LOG_DEBUG("key iterator length", N);
  ASSERT(N == value_last - value_first,
         "The number of items mismatch. # of keys:", N,
         ", # of values:", value_last - value_first);

  // Copy the coordinates to m_coordinate
  base_type::reserve(N);
  CUDA_CHECK(
      cudaMemcpy(coordinate_data(), // dst
                 key_first->data(), // first element of the dereferenced iter.
                 sizeof(coordinate_type) * N * m_coordinate_size, // bytes
                 cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Reserved and copied", N, "x", m_coordinate_size, "coordinates");

  // Insert coordinates
  thrust::counting_iterator<uint32_t> count{0};
  auto insert = detail::insert_coordinate<coordinate_type, map_type,
                                          thrust::counting_iterator<uint32_t>>{
      *m_map,                  // map
      const_coordinate_data(), // coordinates,
      value_first,             // iter begin
      m_coordinate_size};

  thrust::device_vector<bool> success(N);
  m_valid_index.resize(N);
  thrust::sequence(thrust::device, m_valid_index.begin(), m_valid_index.end());

  // Insert coordinates
  thrust::transform(count, count + N, success.begin(), insert);

  // Valid row index
  auto valid_begin = thrust::make_zip_iterator(
      thrust::make_tuple(success.begin(), m_valid_index.begin()));
  size_type const number_of_valid =
      thrust::remove_if(thrust::device, valid_begin,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            success.end(), m_valid_index.end())),
                        detail::is_first<false>()) -
      valid_begin;
  m_valid_index.resize(number_of_valid);
}

/*
 * @brief given a key iterator begin-end pair find all valid keys and its
 * index.
 *
 * @return a pair of (valid index, query value) vectors.
 */
template <typename coordinate_type, typename MapAllocator,
          typename CoordinateAllocator, typename KernelMapAllocator>
std::pair<thrust::device_vector<uint32_t>, thrust::device_vector<uint32_t>>
CoordinateMapGPU<coordinate_type, MapAllocator, CoordinateAllocator,
                 KernelMapAllocator>::find(coordinate_iterator<coordinate_type>
                                               key_first,
                                           coordinate_iterator<coordinate_type>
                                               key_last) const {
  size_type N = key_last - key_first;

  LOG_DEBUG(N, "queries for find.");
  auto const find_functor = detail::find_coordinate<coordinate_type, map_type>(
      *m_map, key_first->data(), m_unused_element, m_coordinate_size);
  LOG_DEBUG("Find functor initialized.");
  auto const invalid_functor =
      detail::is_unused_pair<coordinate_type, mapped_type>(m_unused_element);
  LOG_DEBUG("Valid functor initialized.");

  thrust::counting_iterator<index_type> index{0};
  device_index_vector_type input_index(N);
  device_index_vector_type results(N);
  LOG_DEBUG("Initialized functors.");
  thrust::sequence(thrust::device, input_index.begin(), input_index.end());
  thrust::transform(thrust::device, index, index + N, results.begin(),
                    find_functor);

  size_type const number_of_valid =
      thrust::remove_if(thrust::device,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            input_index.begin(), results.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            input_index.end(), results.end())),
                        invalid_functor) -
      thrust::make_zip_iterator(
          thrust::make_tuple(input_index.begin(), results.begin()));
  LOG_DEBUG("Number of valid", number_of_valid);
  input_index.resize(number_of_valid);
  results.resize(number_of_valid);

  return std::make_pair(input_index, results);
}

/*
 * @brief given a key iterator begin-end pair find all valid keys and its
 * index.
 *
 * @return a pair of (valid index, query value) vectors.
 */
template <typename coordinate_type, typename MapAllocator,
          typename CoordinateAllocator, typename KernelMapAllocator>
CoordinateMapGPU<coordinate_type, MapAllocator, CoordinateAllocator,
                 KernelMapAllocator>
CoordinateMapGPU<coordinate_type, MapAllocator, CoordinateAllocator,
                 KernelMapAllocator>::stride(stride_type const &stride) const {

  // Over estimate the reserve size to be size();
  size_type const N = size();

  self_type stride_map(
      N, m_coordinate_size, m_hashtable_occupancy,
      detail::stride_tensor_stride(base_type::m_tensor_stride, stride),
      base_type::m_allocator);

  // stride coordinates
  thrust::counting_iterator<uint32_t> count_begin{0};
  thrust::for_each(
      count_begin, count_begin + N,
      detail::stride_copy<coordinate_type, index_type>(
          m_coordinate_size,
          thrust::raw_pointer_cast(stride_map.m_device_tensor_stride.data()),
          thrust::raw_pointer_cast(m_valid_index.data()),
          const_coordinate_data(), //
          stride_map.coordinate_data()));

  thrust::device_vector<bool> success(N);
  auto &stride_valid_index = stride_map.m_valid_index;
  stride_valid_index.resize(N);
  thrust::sequence(thrust::device, stride_valid_index.begin(),
                   stride_valid_index.end());

  // Insert coordinates
  auto insert = detail::insert_coordinate<coordinate_type, map_type,
                                          index_type *>{
      *stride_map.m_map,                                   // map
      stride_map.const_coordinate_data(),                  // coordinates,
      thrust::raw_pointer_cast(stride_valid_index.data()), // iter begin
      m_coordinate_size};
  thrust::transform(count_begin, count_begin + N, success.begin(), insert);

  // Valid row index
  auto valid_begin = thrust::make_zip_iterator(
      thrust::make_tuple(success.begin(), stride_valid_index.begin()));
  size_type const number_of_valid =
      thrust::remove_if(thrust::device, valid_begin,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            success.end(), stride_valid_index.end())),
                        detail::is_first<false>()) -
      valid_begin;
  stride_valid_index.resize(number_of_valid);

  thrust::for_each(count_begin, count_begin + number_of_valid,
                   detail::update_value<coordinate_type, map_type>{
                       *stride_map.m_map, stride_map.const_coordinate_data(),
                       thrust::raw_pointer_cast(stride_valid_index.data()),
                       m_coordinate_size});

  return stride_map;
}

namespace detail {

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void count_kernel(map_type const __restrict__ in_map,        //
                             map_type const __restrict__ out_map,       //
                             size_type const num_map_values_per_thread, //
                             size_type const num_threads,               //
                             gpu_kernel_region<coordinate_type> kernel, //
                             index_type *__restrict__ p_count_per_thread) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;

  size_type coordinate_size = kernel.coordinate_size();
  size_type volume = kernel.volume();

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_tensor_stride = sh_size;
  size_type *sh_kernel_size   = sh_tensor_stride + coordinate_size;
  size_type *sh_dilation      = sh_kernel_size   + coordinate_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_dilation + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate +                   tx  * coordinate_size;
  coordinate_type *sh_lb  = sh_coordinate + (1 * blockDim.x + tx) * coordinate_size;
  coordinate_type *sh_ub  = sh_coordinate + (2 * blockDim.x + tx) * coordinate_size;
  // clang-format on

  auto const equal = out_map.get_key_equal();

  // kernel_maps
  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = kernel.tensor_stride()[i];
    sh_kernel_size[i] = kernel.kernel_size()[i];
    sh_dilation[i] = kernel.dilation()[i];
  }

  __syncthreads();

  // clang-format off
  auto const unused_key = out_map.get_unused_key();
  auto const max_index = umin((x + 1) * num_map_values_per_thread, out_map.capacity());
  // iterate over values
  size_type count = 0;
  for (index_type value_index = x * num_map_values_per_thread;
       value_index < max_index;
       ++value_index) {
    typename map_type::value_type const &out_value = out_map.data()[value_index];
    // clang-format on
    if (!equal(out_value.first, unused_key)) {
      // set bounds for the valid keys
      kernel.set_bounds(out_value.first.data(), sh_lb, sh_ub, sh_tmp);
      for (auto const &coordinate : kernel) {
        if (in_map.find(coordinate) != in_map.end()) {
          ++count;
        }
      }
    }
  }

  if (x < num_threads)
    p_count_per_thread[x] = count;
}

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void preallocated_kernel_map_iteration(
    map_type const __restrict__ in_map,                  //
    map_type const __restrict__ out_map,                 //
    size_type const num_map_values_per_thread,           //
    size_type const num_threads,                         //
    gpu_kernel_region<coordinate_type> kernel,           //
    index_type const *inclusive_count_cumsum_per_thread, //
    index_type *__restrict__ p_kernels,                  //
    index_type *__restrict__ p_in_maps,                  //
    index_type *__restrict__ p_out_maps) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;

  size_type coordinate_size = kernel.coordinate_size();
  size_type volume = kernel.volume();

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_tensor_stride = sh_size;
  size_type *sh_kernel_size   = sh_tensor_stride + coordinate_size;
  size_type *sh_dilation      = sh_kernel_size   + coordinate_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_dilation + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate +                   tx  * coordinate_size;
  coordinate_type *sh_lb  = sh_coordinate + (1 * blockDim.x + tx) * coordinate_size;
  coordinate_type *sh_ub  = sh_coordinate + (2 * blockDim.x + tx) * coordinate_size;
  // clang-format on

  auto const equal = out_map.get_key_equal();

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = kernel.tensor_stride()[i];
    sh_kernel_size[i] = kernel.kernel_size()[i];
    sh_dilation[i] = kernel.dilation()[i];
  }

  __syncthreads();

  if (x >= num_threads) return;

  // clang-format off
  auto const unused_key = out_map.get_unused_key();
  auto const max_index = umin((x + 1) * num_map_values_per_thread, out_map.capacity());

  // iterate over values
  auto kernel_map_index = (x < 1) ? 0 : inclusive_count_cumsum_per_thread[x - 1];
  index_type kernel_index = 0;
  for (index_type value_index = x * num_map_values_per_thread;
       value_index < max_index;
       ++value_index) {
    typename map_type::value_type const &out_value = out_map.data()[value_index];
    // clang-format on
    if (!equal(out_value.first, unused_key)) {
      // set bounds for the valid keys
      kernel.set_bounds(out_value.first.data(), sh_lb, sh_ub, sh_tmp);
      kernel_index = 0;
      for (auto const &coordinate : kernel) {
        auto const &in_result = in_map.find(coordinate);
        if (in_result != in_map.end()) {
          // insert to
          p_kernels[kernel_map_index] = kernel_index;
          p_in_maps[kernel_map_index] = (*in_result).second;
          p_out_maps[kernel_map_index] = out_value.second;
          ++kernel_map_index;
        }
        ++kernel_index;
      }
    }
  }
}

} // namespace detail

template <typename coordinate_type, typename MapAllocator,
          typename CoordinateAllocator, typename KernelMapAllocator>
CoordinateMapGPU<coordinate_type, MapAllocator, CoordinateAllocator,
                 KernelMapAllocator>::kernel_map_type
CoordinateMapGPU<coordinate_type, MapAllocator, CoordinateAllocator,
                 KernelMapAllocator>::
    kernel_map(self_type const &out_coordinate_map,
               gpu_kernel_region<coordinate_type> const &kernel,
               uint32_t num_map_values_per_thread, uint32_t thread_dim) const {
  // Over estimate the reserve size to be size();
  size_type out_capacity = out_coordinate_map.m_map->capacity();
  size_type kernel_volume = kernel.volume();

  // clang-format off
  // (THREAD * 3 * D +  3 * D + 2 * K) * 4
  uint32_t shared_memory_size_in_bytes =
      3 * m_coordinate_size * sizeof(index_type) + // stride, kernel, dilation
      3 * thread_dim * m_coordinate_size * sizeof(coordinate_type); // tmp, lb, ub
  // clang-format on
  size_type const num_threads =
      (out_capacity + num_map_values_per_thread - 1) /
      num_map_values_per_thread;
  auto const block_dim = GET_BLOCKS(num_threads, thread_dim);

  LOG_DEBUG("block dim", block_dim);
  LOG_DEBUG("out_coordinate_map capacity", out_coordinate_map.capacity());
  LOG_DEBUG("shared_memory size", shared_memory_size_in_bytes);
  LOG_DEBUG("threads dim", thread_dim);
  LOG_DEBUG("num threads", num_threads);

  index_type *d_p_count_per_thread =
      m_kernel_map_allocator.allocate(num_threads);

  // clang-format off
  // Initialize count per thread
  detail::count_kernel<coordinate_type, size_type, index_type, map_type>
      <<<block_dim, thread_dim, shared_memory_size_in_bytes>>>(
          *m_map,                    //
          *out_coordinate_map.m_map, //
          num_map_values_per_thread, //
          num_threads,               //
          kernel,                    //
          d_p_count_per_thread);
  // clang-format on
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("count_kernel finished");

  thrust::inclusive_scan(thrust::device, d_p_count_per_thread,
                         d_p_count_per_thread + num_threads,
                         d_p_count_per_thread);

  index_type num_kernel_map; // type following the kernel map allocator
  CUDA_CHECK(cudaMemcpy(&num_kernel_map, d_p_count_per_thread + num_threads - 1,
                        sizeof(index_type), cudaMemcpyDeviceToHost));

  // set kernel map
  LOG_DEBUG("Found", num_kernel_map, "kernel map elements.");

  kernel_map_type kernel_map(num_kernel_map, m_kernel_map_allocator);
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Allocated kernel_map.");

  detail::preallocated_kernel_map_iteration<coordinate_type, size_type,
                                            index_type, map_type>
      <<<block_dim, thread_dim, shared_memory_size_in_bytes>>>(
          *m_map,                     //
          *out_coordinate_map.m_map,  //
          num_map_values_per_thread,  //
          num_threads,                //
          kernel,                     //
          d_p_count_per_thread,       //
          kernel_map.kernels.begin(), //
          kernel_map.in_maps.begin(), //
          kernel_map.out_maps.begin());

  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Preallocated kernel map done");

  kernel_map.decompose();
  m_kernel_map_allocator.deallocate(d_p_count_per_thread, num_threads);
  LOG_DEBUG("cudaFree");

  return kernel_map;
}

// Template instantiation
template class CoordinateMapGPU<default_types::dcoordinate_type>;
// Insert arg templates
using citer32 = coordinate_iterator<default_types::dcoordinate_type>;
template void CoordinateMapGPU<default_types::dcoordinate_type>::insert<
    thrust::counting_iterator<default_types::index_type>>(
    citer32,                                              // key bein
    citer32,                                              // key end
    thrust::counting_iterator<default_types::index_type>, // value begin
    thrust::counting_iterator<default_types::index_type>  // value end
);

} // namespace minkowski
