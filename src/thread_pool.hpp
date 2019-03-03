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
  Triplets
  operator()(const Coord<D, int> out_coord, const Arr<D, int> &in_pixel_dists,
             const Arr<D, int> &kernel_size, const Arr<D, int> &dilations,
             const int region_type, const Itype *offsets_data,
             const int offsets_size, const int out_coord_index,
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
          const Arr<D, int> &in_pixel_dists, const Arr<D, int> &kernel_size,
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
