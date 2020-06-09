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
          typename CoordinateAllocator, typename IndexAllocator>
template <typename mapped_iterator>
void CoordinateMapGPU<
    coordinate_type, MapAllocator, CoordinateAllocator,
    IndexAllocator>::insert(coordinate_iterator<coordinate_type> key_first,
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
  LOG_DEBUG("Reserved and copied", N, "x", m_coordinate_size);

  // Insert coordinates
  thrust::counting_iterator<uint32_t> count{0};
  auto insert = detail::insert_coordinate<coordinate_type, map_type,
                                          thrust::counting_iterator<uint32_t>>{
      *m_map,            // map
      coordinate_data(), // coordinates,
      value_first,       // iter begin
      m_coordinate_size};
  thrust::for_each(count, count + N, insert);
}

/*
 * @brief given a key iterator begin-end pair find all valid keys and its
 * index.
 *
 * @return a pair of (valid index, query value) vectors.
 */
template <typename coordinate_type, typename MapAllocator,
          typename CoordinateAllocator, typename IndexAllocator>
thrust::pair<thrust::device_vector<uint32_t>, thrust::device_vector<uint32_t>>
CoordinateMapGPU<
    coordinate_type, MapAllocator, CoordinateAllocator,
    IndexAllocator>::find(coordinate_iterator<coordinate_type> key_first,
                          coordinate_iterator<coordinate_type> key_last) const {
  size_type N = key_last - key_first;

  // reserve the result slots
  index_vector_type valid_query_index, query_result;
  valid_query_index.reserve(N);
  query_result.reserve(N);

  LOG_DEBUG(N, "queries for find.")
  auto const find_functor = detail::find_coordinate<coordinate_type, map_type>(
      *m_map, key_first->data(), m_unused_element, m_coordinate_size);
  LOG_DEBUG("Find functor initialized.")
  auto const invalid_functor =
      detail::is_invalid_pair<coordinate_type, mapped_type>(m_unused_element);
  LOG_DEBUG("Valid functor initialized.")

  thrust::counting_iterator<index_type> index{0};
  query_result_type input_index(N);
  query_result_type results(N);
  LOG_DEBUG("Initialized functors.")
  thrust::sequence(input_index.begin(), input_index.end());
  thrust::transform(thrust::device, index, index + N, results.begin(),
                    find_functor);

  size_type const number_of_valid =
      thrust::remove_if(thrust::device,
                        thrust::make_zip_iterator(thrust::make_tuple(input_index.begin(), results.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(input_index.end(), results.end())),
                        invalid_functor) -
      thrust::make_zip_iterator(thrust::make_tuple(input_index.begin(), results.begin()));
  LOG_DEBUG("Number of valid", number_of_valid);
  input_index.resize(number_of_valid);
  results.resize(number_of_valid);

  return thrust::make_pair(input_index, results);
}

// Template instantiation
template class CoordinateMapGPU<int32_t>;
// Insert arg templates
using citer32 = coordinate_iterator<int32_t>;
template void
    CoordinateMapGPU<int32_t>::insert<thrust::counting_iterator<uint32_t>>(
        citer32, citer32, thrust::counting_iterator<uint32_t>,
        thrust::counting_iterator<uint32_t>);

} // namespace minkowski
