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
__device__ void device_atomic_addition(Dtype *dst, const Dtype *src,
                                       int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    atomicAdd(dst + i, src[i]);
}

template <typename Dtype>
__device__ void device_memcpy(Dtype *dst, const Dtype *src, int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    dst[i] = src[i];
}

template <typename Dtype, typename Itype>
__global__ void add_in_out_map(const int n, const Dtype *in_feat,
                               Dtype *out_feat, const int nchannel,
                               const Itype *in_map, const Itype *out_map) {
  CUDA_KERNEL_LOOP(index, n) {
    device_atomic_addition<Dtype>(&out_feat[out_map[index] * nchannel],
                                  &in_feat[in_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype, typename Itype>
__global__ void copy_in_out_map(const int n, const Dtype *in_feat,
                                Dtype *out_feat, const int nchannel,
                                const Itype *in_map, const Itype *out_map) {
  CUDA_KERNEL_LOOP(index, n) {
    device_memcpy<Dtype>(&out_feat[out_map[index] * nchannel],
                         &in_feat[in_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype, typename Itype>
void UnionForwardKernelGPU(const vector<Dtype *> d_in_feats, Dtype *d_out_feat,
                           const int nchannel, const pInOutMaps<Itype> &in_maps,
                           const pInOutMaps<Itype> &out_maps,
                           cudaStream_t stream) {

  for (size_t k = 0; k < in_maps.size(); k++) {
    const size_t nnz = in_maps[k].size();
    add_in_out_map<Dtype, Itype>
        <<<GET_BLOCKS(nnz), CUDA_NUM_THREADS, 0, stream>>>(
            nnz, d_in_feats[k], d_out_feat, nchannel, in_maps[k].data(),
            out_maps[k].data());
  }
}

template <typename Dtype, typename Itype>
void UnionBackwardKernelGPU(vector<Dtype *> d_grad_in_feats,
                            const Dtype *d_grad_out_feat, int nchannel,
                            const pInOutMaps<Itype> &in_maps,
                            const pInOutMaps<Itype> &out_maps,
                            cudaStream_t stream) {

  for (size_t k = 0; k < in_maps.size(); k++) {
    const int nnz = in_maps[k].size();
    copy_in_out_map<Dtype, Itype>
        <<<GET_BLOCKS(nnz), CUDA_NUM_THREADS, 0, stream>>>(
            nnz, d_grad_out_feat, d_grad_in_feats[k], nchannel,
            out_maps[k].data(), in_maps[k].data());
  }
}

template void UnionForwardKernelGPU<float, int32_t>(
    const vector<float *> d_in_feats, float *d_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);

template void UnionBackwardKernelGPU<float, int32_t>(
    vector<float *> d_grad_in_feats, const float *d_grad_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);

template void UnionForwardKernelGPU<double, int32_t>(
    const vector<double *> d_in_feats, double *d_out_feat, int nchannel,
    const pInOutMaps<int32_t> &in_maps, const pInOutMaps<int32_t> &out_maps,
    cudaStream_t stream);

template void UnionBackwardKernelGPU<double, int32_t>(
    vector<double *> d_grad_in_feats, const double *d_grad_out_feat,
    int nchannel, const pInOutMaps<int32_t> &in_maps,
    const pInOutMaps<int32_t> &out_maps, cudaStream_t stream);
