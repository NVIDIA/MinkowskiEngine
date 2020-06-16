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
#include "kernel_region.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>

#include <torch/extension.h>

namespace minkowski {

using coordinate_type = int32_t;
using index_type = default_types::index_type;
using size_type = default_types::size_type;

/*
 * The number of threads must be > coordinate_size
 */
__global__ void kernel_region_iterator_test(
    coordinate_type const *__restrict__ p_coordinate,
    size_type number_of_coordinates,
    index_type const *__restrict__ tensor_stride,
    index_type const *__restrict__ kernel_size,
    index_type const *__restrict__ dilation, index_type coordinate_size,
    coordinate_type *__restrict__ p_return_coordinates) {
  extern __shared__ coordinate_type sh_coordinate[];

  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;

  coordinate_type *sh_tmp = sh_coordinate;
  coordinate_type *sh_lb = sh_coordinate + CUDA_NUM_THREADS * coordinate_size;
  coordinate_type *sh_ub = sh_lb + coordinate_size;

  index_type *sh_index =
      reinterpret_cast<index_type *>(sh_ub + coordinate_size);
  index_type *sh_tensor_stride = sh_index;
  index_type *sh_kernel_size = sh_index + coordinate_size;
  index_type *sh_dilation = sh_index + 2 * coordinate_size;

  index_type a;
  if (tx < coordinate_size - 1) {
    sh_tensor_stride[tx] = tensor_stride[tx];
    sh_kernel_size[tx] = kernel_size[tx];
    sh_dilation[tx] = dilation[tx];
  }
  __syncthreads();
  if (x >= number_of_coordinates)
    return;

  auto region = kernel_region<coordinate_type, HYPER_CUBE>(
      coordinate_size, sh_tensor_stride, sh_kernel_size, sh_dilation);
  index_type kernel_volume = region.volume();

  // iterate and copy
  index_type out_index = x * kernel_volume;
  region.set_bounds(&p_coordinate[x * coordinate_size], sh_lb, sh_ub,
                    &sh_tmp[tx * coordinate_size]);
  for (auto const &coordinate : region) {
    for (index_type i = 0; i < coordinate_size; ++i) {
      p_return_coordinates[out_index * coordinate_size + i] = coordinate[i];
    }
    ++out_index;
  }
}

at::Tensor region_iterator_test(const torch::Tensor &coordinates,
                                const torch::Tensor &kernel_size) {
  // Create TensorArgs. These record the names and positions of each tensor as
  // parameters.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);
  torch::TensorArg arg_kernel_size(kernel_size, "kernel_size", 1);

  torch::CheckedFrom c = "region_iterator_test";
  torch::checkContiguous(c, arg_coordinates);
  torch::checkContiguous(c, arg_kernel_size);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_kernel_size, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_kernel_size.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);
  torch::checkDim(c, arg_kernel_size, 1);

  auto const N = (index_type)coordinates.size(0);
  auto const coordinate_size = (index_type)coordinates.size(1);
  coordinate_type *p_coordinate = coordinates.data_ptr<coordinate_type>();
  coordinate_type *p_kernel_size = kernel_size.data_ptr<coordinate_type>();

  thrust::device_vector<index_type> thrust_tensor_stride(coordinate_size - 1);
  thrust::device_vector<index_type> thrust_kernel_size(coordinate_size - 1);
  thrust::device_vector<index_type> thrust_dilation(coordinate_size - 1);

  index_type kernel_volume = 1;
  for (index_type i = 0; i < coordinate_size - 1; ++i) {
    kernel_volume *= p_kernel_size[i];
  }

  LOG_DEBUG("kernel_volume", kernel_volume);
  for (index_type i = 0; i < coordinate_size - 1; ++i) {
    thrust_tensor_stride[i] = 1;
    thrust_kernel_size[i] = p_kernel_size[i];
    thrust_dilation[i] = 1;
  }
  CUDA_CHECK(cudaStreamSynchronize(0));

  LOG_DEBUG("initialize vectors");

  torch::Tensor out_coordinates =
      torch::empty({N * kernel_volume, coordinate_size}, coordinates.options());

  uint32_t shared_memory_size_in_bytes =
      (2 + CUDA_NUM_THREADS) * coordinate_size * sizeof(coordinate_type) +
      3 * coordinate_size * sizeof(index_type);

  kernel_region_iterator_test<<<GET_BLOCKS(N), CUDA_NUM_THREADS,
                                shared_memory_size_in_bytes>>>(
      p_coordinate, //
      N,            //
      thrust::raw_pointer_cast(thrust_tensor_stride.data()),
      thrust::raw_pointer_cast(thrust_kernel_size.data()),
      thrust::raw_pointer_cast(thrust_dilation.data()), //
      coordinate_size,                                  //
      out_coordinates.data_ptr<coordinate_type>());

  LOG_DEBUG("End call");
  CUDA_CHECK(cudaStreamSynchronize(0));

  return out_coordinates;
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("region_iterator_test", &minkowski::region_iterator_test,
        "Minkowski Engine region iterator test");
}
