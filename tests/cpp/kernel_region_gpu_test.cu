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
    size_type number_of_coordinates, //
    gpu_kernel_region<coordinate_type> kernel,
    coordinate_type *__restrict__ p_return_coordinates) {
  extern __shared__ coordinate_type sh_coordinate[];

  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;

  size_type coordinate_size = kernel.coordinate_size();
  size_type volume = kernel.volume();

  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  coordinate_type *sh_lb =
      sh_coordinate + (CUDA_NUM_THREADS + tx) * coordinate_size;
  coordinate_type *sh_ub =
      sh_coordinate + (2 * CUDA_NUM_THREADS + tx) * coordinate_size;

  index_type *sh_index = reinterpret_cast<index_type *>(
      sh_coordinate + 3 * CUDA_NUM_THREADS * coordinate_size);
  index_type *sh_tensor_stride = sh_index;
  index_type *sh_kernel_size = sh_index + coordinate_size;
  index_type *sh_dilation = sh_index + 2 * coordinate_size;

  if (tx < coordinate_size - 1) {
    sh_tensor_stride[tx] = kernel.tensor_stride()[tx];
    sh_kernel_size[tx] = kernel.kernel_size()[tx];
    sh_dilation[tx] = kernel.dilation()[tx];
  }
  __syncthreads();
  if (x >= number_of_coordinates)
    return;

  // iterate and copy
  index_type out_index = x * kernel.volume();
  kernel.set_bounds(&p_coordinate[x * coordinate_size], sh_lb, sh_ub, sh_tmp);
  for (auto const &coordinate : kernel) {
    for (index_type i = 0; i < coordinate_size; ++i) {
      p_return_coordinates[out_index * coordinate_size + i] = coordinate[i];
    }
    ++out_index;
  }
}

at::Tensor region_iterator_test(const torch::Tensor &coordinates,
                                const torch::Tensor &th_kernel_size) {
  // Create TensorArgs. These record the names and positions of each tensor as
  // parameters.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);
  torch::TensorArg arg_kernel_size(th_kernel_size, "kernel_size", 1);

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
  coordinate_type *p_kernel_size = th_kernel_size.data_ptr<coordinate_type>();

  default_types::stride_type tensor_stride(coordinate_size - 1);
  default_types::stride_type kernel_size(coordinate_size - 1);
  default_types::stride_type dilation(coordinate_size - 1);

  for (index_type i = 0; i < coordinate_size - 1; ++i) {
    tensor_stride[i] = 1;
    kernel_size[i] = p_kernel_size[i];
    dilation[i] = 1;
  }

  auto cpu_kernel = cpu_kernel_region<coordinate_type>(
      RegionType::HYPER_CUBE, coordinate_size, tensor_stride.data(),
      kernel_size.data(), dilation.data());
  auto kernel = gpu_kernel_region<coordinate_type>(cpu_kernel.to_gpu());

  LOG_DEBUG("initialize vectors");

  torch::Tensor out_coordinates = torch::empty(
      {N * kernel.volume(), coordinate_size}, coordinates.options());

  uint32_t shared_memory_size_in_bytes =
      3 * CUDA_NUM_THREADS * coordinate_size * sizeof(coordinate_type) +
      3 * coordinate_size * sizeof(index_type);

  kernel_region_iterator_test<<<GET_BLOCKS(N, CUDA_NUM_THREADS),
                                CUDA_NUM_THREADS,
                                shared_memory_size_in_bytes>>>(
      p_coordinate, //
      N,            //
      kernel,       //
      out_coordinates.data_ptr<coordinate_type>());

  LOG_DEBUG("End call");
  CUDA_CHECK(cudaStreamSynchronize(0));

  return out_coordinates;
}

std::tuple<std::pair<cpu_in_maps, cpu_out_maps>, size_type, double>
kernel_map_test(const torch::Tensor &in_coordinates,
                const torch::Tensor &out_coordinates,
                const torch::Tensor &kernel_size,
                uint32_t occupancy, //
                uint32_t thread_dim) {
  // Create TensorArgs. These record the names and positions of each tensor as
  // parameters.
  torch::TensorArg arg_in_coordinates(in_coordinates, "coordinates", 0);
  torch::TensorArg arg_out_coordinates(out_coordinates, "coordinates", 1);
  torch::TensorArg arg_kernel_size(kernel_size, "kernel_size", 2);

  torch::CheckedFrom c = "region_iterator_test";
  torch::checkContiguous(c, arg_in_coordinates);
  torch::checkContiguous(c, arg_out_coordinates);
  torch::checkContiguous(c, arg_kernel_size);
  // must match coordinate_type
  torch::checkScalarType(c, arg_in_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_out_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_kernel_size, torch::kInt);
  torch::checkBackend(c, arg_in_coordinates.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_out_coordinates.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_kernel_size.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_in_coordinates, 2);
  torch::checkDim(c, arg_out_coordinates, 2);
  torch::checkDim(c, arg_kernel_size, 1);

  auto const N_in = (index_type)in_coordinates.size(0);
  auto const D = (index_type)in_coordinates.size(1);

  auto const N_out = (index_type)out_coordinates.size(0);
  auto const D_out = (index_type)out_coordinates.size(1);

  ASSERT(D == D_out, "dimension mismatch");

  coordinate_type const *ptr_in = in_coordinates.data_ptr<coordinate_type>();
  coordinate_type const *ptr_out = out_coordinates.data_ptr<coordinate_type>();

  CoordinateMapGPU<coordinate_type> in_map{N_in, D, occupancy};
  CoordinateMapGPU<coordinate_type> out_map{N_out, D, occupancy};

  auto in_coordinate_range = coordinate_range<coordinate_type>(N_in, D, ptr_in);
  in_map.insert<false>(in_coordinate_range.begin(), // key begin
                       in_coordinate_range.end());  // key end

  auto out_coordinate_range =
      coordinate_range<coordinate_type>(N_out, D, ptr_out);
  out_map.insert<false>(out_coordinate_range.begin(), // key begin
                        out_coordinate_range.end());  // key end

  LOG_DEBUG("coordinate initialization");

  // Kernel region
  coordinate_type *p_kernel_size = kernel_size.data_ptr<coordinate_type>();
  default_types::stride_type tensor_stride;
  default_types::stride_type s_kernel_size;
  default_types::stride_type dilation;
  for (index_type i = 0; i < D - 1; ++i) {
    tensor_stride.push_back(1);
    s_kernel_size.push_back(p_kernel_size[i]);
    dilation.push_back(1);
  }

  auto region = cpu_kernel_region<coordinate_type>(
      RegionType::HYPER_CUBE, D, tensor_stride.data(), s_kernel_size.data(),
      dilation.data());
  LOG_DEBUG("cpu_kernel_region initialized with volume", region.volume());
  region.to_gpu();
  auto gpu_region = gpu_kernel_region<coordinate_type>(region);
  LOG_DEBUG("gpu_kernel_region initialization");

  timer t;
  t.tic();
  auto kernel_map = in_map.kernel_map(
      out_map, gpu_region, CUDAKernelMapMode::SPEED_OPTIMIZED, thread_dim);
  double k_time = t.toc();

  const auto volume = region.volume();
  LOG_DEBUG("kernel_map done");

  cpu_in_maps in_maps(volume);
  cpu_in_maps out_maps(volume);
  for (index_type i = 0; i < volume; ++i) {
    size_type size = kernel_map.kernels.size(i);
    LOG_DEBUG("kernel index", i, "/", volume, "with size", size);
    in_maps[i].resize(size);
    out_maps[i].resize(size);

    if (size > 0) {
      cudaMemcpy(in_maps[i].data(), //
                 kernel_map.in_maps.begin(i), sizeof(index_type) * size,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(out_maps[i].data(), //
                 kernel_map.out_maps.begin(i), sizeof(index_type) * size,
                 cudaMemcpyDeviceToHost);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(0));

  return std::make_tuple(std::make_pair(in_maps, out_maps), out_map.size(),
                         k_time);
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("region_iterator_test", &minkowski::region_iterator_test,
        "Minkowski Engine region iterator test");

  m.def("kernel_map_test", &minkowski::kernel_map_test,
        "Minkowski Engine kernel map test");
}
