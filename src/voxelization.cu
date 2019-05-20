/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do
 *  so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <exception>
#include <stdexcept>
#include <typeinfo>

#include "gpu.cuh"
#include "voxelization.cuh"

/*
 * If the comparator is called, it means that there are duplicate labels.
 * return ignore_label always.
 */
template <typename LabelType, typename Itype>
struct ConsolidateLabel
    : public thrust::binary_function<const thrust::tuple<LabelType, Itype> &,
                                     const thrust::tuple<LabelType, Itype> &,
                                     thrust::tuple<LabelType, Itype>> {
  const LabelType ignore_label;

  ConsolidateLabel(LabelType _ignore_label) : ignore_label(_ignore_label) {}

  __host__ __device__ thrust::tuple<LabelType, Itype>
  operator()(const thrust::tuple<LabelType, Itype> &x,
             const thrust::tuple<LabelType, Itype> &y) const {
    thrust::tuple<LabelType, Itype> tmp;
    thrust::get<0>(tmp) = ignore_label;
    thrust::get<1>(tmp) = thrust::get<1>(x);
    return tmp;
  }
};

typedef thrust::tuple<uint64_t, int> tuple_t;
struct tupleEqual {
  __host__ __device__ bool operator()(tuple_t x, tuple_t y) {
    return ((x.get<0>() == y.get<0>()) && (x.get<1>() == y.get<1>()));
  }
};

/*
 * Given the unique hash keys of the coordinates and their labels, return
 * indices of coords and corresponding labels.
 * For feature, use the return_key_indices.
 *
 * ignore_label is by default 255
 * TODO: mode is either unique or invalid.
 *   Unique will return the first label in the sorted key,
 *   Invalid will return invalid if there are duplicates.
 * TODO: has_label
 */
int sparse_voxelization(uint64_t *keys, int *labels, int **return_key_indices,
                        int **return_labels, int n, int ignore_label,
                        int has_label) {
  try {
    thrust::device_vector<uint64_t> d_key(keys, keys + n);
    thrust::device_vector<int> d_label(labels, labels + n);
    thrust::device_vector<int> d_return_key_indices(n);

    // Index used to recover the original index after sparse voxelization.
    thrust::device_vector<int> d_index(n);
    thrust::sequence(d_index.begin(), d_index.end());

    // 1. `thrust::sort_by_key` to sort feat, label by the coord hash

    // Returns sorted d_key and d_index sorted by the d_key ordering
    thrust::sort_by_key(thrust::device, d_key.begin(), d_key.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_label.begin(), d_index.begin())));

    // 2. zip the sorted label and the sorted key and use `thrust::unique` to
    // remove duplicates.
    thrust::device_vector<int>::iterator end =
        thrust::unique_by_key(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(
                                  d_key.begin(), d_label.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(
                                  d_key.end(), d_label.end())),
                              d_index.begin())
            .second;

    // End tuple ((d_key, d_label), d_index).
    int reduced_n = end - d_index.begin();

    // 3. Use the `thrust::reduce_by_key` with custom comparator that returns
    // 255 (ignore_label) always. (If it has to be compared, it means there are
    // at least two labels for one key).
    thrust::device_vector<uint64_t> d_key_out(reduced_n);
    thrust::device_vector<int> d_label_out(reduced_n);
    thrust::device_vector<int> d_index_out(reduced_n);
    thrust::equal_to<uint64_t> equal_to;
    ConsolidateLabel<int, int> consolidate_label(ignore_label);
    thrust::device_vector<uint64_t>::iterator final_end =
        thrust::reduce_by_key(thrust::device,
            d_key.begin(), d_key.begin() + reduced_n, // Key
            thrust::make_zip_iterator(
                thrust::make_tuple(d_label.begin(), d_index.begin())), // Val
            d_key_out.begin(), // Key out
            thrust::make_zip_iterator(thrust::make_tuple(
                d_label_out.begin(), d_index_out.begin())), // Val out
            equal_to, consolidate_label)
            .first;
    int final_n = final_end - d_key_out.begin();

    int *h_label = new int[final_n]();
    int *h_index = new int[final_n]();
    CUDA_CHECK(cudaMemcpy(h_label, thrust::raw_pointer_cast(d_label_out.data()),
                          final_n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_index, thrust::raw_pointer_cast(d_index_out.data()),
                          final_n * sizeof(int), cudaMemcpyDeviceToHost));
    *return_key_indices = h_index;
    *return_labels = h_label;
    return final_n;
  } catch (thrust::system_error e) {
    throw std::runtime_error(Formatter()
                             << "Thrust error: " << e.what() << " at "
                             << __FILE__ << ":" << __LINE__);
  }
}

void cuda_thread_exit(void) {
  // printf("Exit the current cuda thread\n");
  CUDA_CHECK(cudaThreadExit());
}
