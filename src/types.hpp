#ifndef TYPES
#define TYPES

#include <functional>
#include <google/dense_hash_map>

// N-Dimensional coordinate + batch index = N + 1
template <uint8_t D, typename Itype> using Coord = std::array<Itype, D + 1>;

// For hashing kernel sizes, strides, and dilations.
template <uint8_t D, typename Itype> using Arr = std::array<Itype, D>;

// Key for InOutMap
// (in_coords_key, out_coords_key, stride hash, kernel size, dilation,
// is_transpose)
using InOutMapKey = std::array<uint64_t, 7>;

// Input index to output index mapping for each spatial kernel
template <typename Itype>
using InOutMapPerKernel = std::vector<std::vector<Itype>>;

template <typename Itype>
using InOutKernelMapPair =
    std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>;

// FNV64-1a
// uint64_t for unsigned long, must use CXX -m64
// WARNING: IType for T must be int32
template <typename T> uint64_t hash_vec(T p) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (uint32_t x : p) {
    hash ^= x;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

struct InOutMapKeyHash {
  uint64_t operator()(InOutMapKey const &p) const {
    return hash_vec<InOutMapKey>(p);
  }
};

// For Used for fast index of coordinate retrieval
template <uint8_t D, typename Itype> struct CoordHash {
  uint64_t operator()(Coord<D, Itype> const &p) const {
    return hash_vec<Coord<D, Itype>>(p);
  }
};

template <uint8_t D, typename Itype> struct ArrHash {
  uint64_t operator()(Arr<D, Itype> const &p) const {
    return hash_vec<Arr<D, Itype>>(p);
  }
};

// Location to index of the feature
template <uint8_t D, typename Itype>
using _CoordsHashMap =
    google::dense_hash_map<Coord<D, Itype>, uint64_t, CoordHash<D, Itype>,
                           std::equal_to<Coord<D, Itype>>>;

template <uint8_t D, typename Itype> class CoordsHashMap {
public:
  _CoordsHashMap<D, Itype> map;
  CoordsHashMap() {
    Coord<D, Itype> empty_key;
    for (int i = 0; i < D + 1; ++i)
      empty_key[i] = -std::numeric_limits<Itype>::max();
    map.set_empty_key(empty_key);
  }
  size_t size() { return map.size(); }
};

// For threaded kernel map
using Triplets = std::vector<std::array<uint32_t, 3>>;

#endif
