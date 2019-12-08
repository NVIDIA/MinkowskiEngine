#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>

#include <iostream>
#include <set>
#include <vector>

#include "region.hpp"
#include "types.hpp"

template <typename Itype>
void stride_ptr(Itype *coord, const vector<Itype> &tensor_strides) {
  for (int i = 0; i < tensor_strides.size(); i++) {
#ifdef BATCH_FIRST
    coord[i + 1] = (coord[i + 1] / tensor_strides[i]) * tensor_stride[i];
#else
    coord[i] = (coord[i] / tensor_strides[i]) * tensor_strides[i];
#endif
  }
}

template <typename Itype>
void stride_copy(Itype *coord_src, Itype *coord_dst,
                 const vector<Itype> &tensor_strides) {
  for (int i = 0; i < tensor_strides.size(); i++) {
#ifdef BATCH_FIRST
    coord_dst[i + 1] =
        (coord_src[i + 1] / tensor_strides[i]) * tensor_stride[i];
#else
    coord_dst[i] = (coord_src[i] / tensor_strides[i]) * tensor_strides[i];
#endif
  }
#ifdef BATCH_FIRST
  coord_dst[0] = coord_src[0];
#else
  coord_dst[tensor_strides.size()] = coord_src[tensor_strides.size()];
#endif
}

template <typename T> uint64_t pointer_hash_vec(T *p, int size) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (int d = 0; d < size; d++) {
    hash ^= p[d];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
};

template <typename Itype> struct PointerCoordHash {
  uint64_t operator()(const Coord<Itype> &c) const {
    return pointer_hash_vec<Itype>(c.ptr, c.size);
  }
};

template <typename T> bool ptr_equal_to(T *l, T *r, int size) {
  bool equal = true;
  int i = 0;
  while (equal && i < size) {
    equal &= l[i] == r[i];
    i++;
  }
  return equal;
}

template <typename Itype> struct PointerEqualTo {
  bool operator()(const Coord<Itype> &lc, Itype *r) const {
    return ptr_equal_to<Itype>(lc.ptr, r, lc.size);
  }

  bool operator()(const Coord<Itype> &lc, const Coord<Itype> &rc) const {
    return (lc.size == rc.size) && ptr_equal_to<Itype>(lc.ptr, rc.ptr, lc.size);
  }
};

using ConcurrentCoordsInnerMap =
    tbb::concurrent_unordered_map<Coord<int>, uint64_t, PointerCoordHash<int>,
                                  PointerEqualTo<int>>;

// Unordered map for coordinates
class ConcurrentCoordsMap {
public:
  vector<int> coords;
  int ncols, nrows;
  int num_threads = -1;

  // unordered map int pointer -> index. Should not be called directly
  ConcurrentCoordsInnerMap map;

  // Defined in the abstract class
  //
  // nrows * ncols in row-major order (1, 2), (3, 4) -> [[1, 2], [3, 4]]
  // std::vector<int> coords;         // explicit ownership of the coordinates.
  // std::vector<int> tensor_strides; // Tensor stride of the current coords map
  // int nrows, ncols;

  void set_threads(int num_threads_);

  // Constructors
  ConcurrentCoordsMap() {}
  ConcurrentCoordsMap(int num_threads_) { set_threads(num_threads_); }
  ConcurrentCoordsMap(int ncols_, const set<int> &batch_indices);

  // Initializations
  //
  // Preferably, use std::move to coords_.
  // returns mapping: out_coord row index to in_coord row index
  pair<vector<int>, set<int>> initialize(vector<int> &&coords_, int nrows_,
                                         int ncols_, bool force_remap = false);
  // Generate strided version of the input coordinate map.
  // returns mapping: out_coord row index to in_coord row index
  ConcurrentCoordsMap stride(const vector<int> &tensor_strides);
  ConcurrentCoordsMap stride_region(const Region &region);
  ConcurrentCoordsMap prune(bool *p_keep, int n);

  // Get the unique coords from the input coordinates and mapping
  void updateUniqueCoords(vector<int> &coords_,
                          const tbb::concurrent_vector<int> &mapping);

  // Generate in-out kernel maps
  InOutMapsPair<int> kernel_map(const ConcurrentCoordsMap &out_coords_map,
                                const Region &region) const;
  InOutMapsPair<int>
  pruned_kernel_map(const ConcurrentCoordsMap &out_coords_map) const;
  InOutMapsPair<int>
  global_reduction_map(const ConcurrentCoordsMap &gout_coords_map) const;
  // Generate stride map
  InOutMapsPair<int> stride_map(const ConcurrentCoordsMap &out_coords_map,
                                const vector<int> &tensor_strides) const;

  // Iterators
  ConcurrentCoordsInnerMap::iterator begin() { return map.begin(); }
  ConcurrentCoordsInnerMap::const_iterator begin() const { return map.begin(); }
  ConcurrentCoordsInnerMap::iterator end() { return map.end(); }
  ConcurrentCoordsInnerMap::const_iterator end() const { return map.end(); }
  ConcurrentCoordsInnerMap::iterator find(const Coord<int> &key) {
    return map.find(key);
  }
  ConcurrentCoordsInnerMap::const_iterator find(const Coord<int> &key) const {
    return map.find(key);
  }

  size_t size() const { return map.size(); }

  uint64_t &operator[](const Coord<int> &coord) { return map[coord]; }

  uint64_t &operator[](int *p) {
    Coord<int> coord(p, ncols);
    return map[coord];
  }

  void print() const;
};
