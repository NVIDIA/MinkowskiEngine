/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "gpu.cuh"
#include "pruning.cuh"

template <typename Dtype>
__device__ void device_memcpy(Dtype *dst, const Dtype *src, int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    dst[i] = src[i];
}

template <typename Dtype, typename Itype>
__global__ void copy_in_out_map(const int n, const Dtype *in_feat,
                                Dtype *out_feat, const int nchannel,
                                const Itype *in_map, const Itype *out_map) {
  CUDA_KERNEL_LOOP(index, n) {
    device_memcpy(&out_feat[out_map[index] * nchannel],
                  &in_feat[in_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype, typename Itype>
void PruningForwardKernelGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                             const int nchannel,
                             const pInOutMaps<Itype> &in_maps,
                             const pInOutMaps<Itype> &out_maps,
                             cudaStream_t stream) {
  const int nnz = in_maps[0].size();

  copy_in_out_map<Dtype, Itype>
      <<<GET_BLOCKS(nnz), CUDA_NUM_THREADS, 0, stream>>>(
          nnz, d_in_feat, d_out_feat, nchannel, in_maps[0].data(),
          out_maps[0].data());
}

template <typename Dtype, typename Itype>
void PruningBackwardKernelGPU(Dtype *d_grad_in_feat,
                              const Dtype *d_grad_out_feat, int nchannel,
                              const pInOutMaps<Itype> &in_maps,
                              const pInOutMaps<Itype> &out_maps,
                              cudaStream_t stream) {
  const int nnz = in_maps[0].size();

  copy_in_out_map<Dtype, Itype>
      <<<GET_BLOCKS(nnz), CUDA_NUM_THREADS, 0, stream>>>(
          nnz, d_grad_out_feat, d_grad_in_feat, nchannel, out_maps[0].data(),
          in_maps[0].data());
}

template void PruningForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, float *d_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);

template void PruningBackwardKernelGPU<float, int32_t>(
    float *d_grad_in_feat, const float *d_grad_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);

template void PruningForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, double *d_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);

template void PruningBackwardKernelGPU<double, int32_t>(
    double *d_grad_in_feat, const double *d_grad_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);
