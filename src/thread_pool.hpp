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
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <array>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "instantiation.hpp"
#include "types.hpp"

template <uint8_t D, typename Itype> class KernelMapFunctor {
public:
  Triplets operator()(const Coord<D, int> out_coord,
                      const Arr<D, int> &in_tensor_strides,
                      const Arr<D, int> &kernel_size,
                      const Arr<D, int> &dilations, const int region_type,
                      const Itype *offsets_data, const int offsets_size,
                      const int out_coord_index,
                      const _CoordsHashMap<D, Itype> &in_coords_hashmap);
};

template <uint8_t D, typename Itype> class CoordsThreadPool {
public:
  size_t nthreads;
  CoordsThreadPool(size_t);
  // template <class F, class... Args>
  // auto enqueue(F &&f, Args &&... args)
  //     -> std::future<typename std::result_of<F(Args...)>::type>;
  std::future<Triplets>
  enqueue(KernelMapFunctor<D, Itype> &f, const Coord<D, int> out_coord,
          const Arr<D, int> &in_tensor_strides, const Arr<D, int> &kernel_size,
          const Arr<D, int> &dilations, const int region_type,
          const Itype *offsets_data, const int offsets_size,
          const int out_coord_index,
          const _CoordsHashMap<D, Itype> &in_coords_hashmap);

  ~CoordsThreadPool();

private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

#endif
