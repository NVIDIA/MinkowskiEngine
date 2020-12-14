#include "coordinate.h"

///*
template <typename T, size_t D>
__device__ __host__ bool Coordinate<T, D>::operator==(const Coordinate<T, D>& rhs) const {
    bool equal = true;
#pragma unroll 1
    for (size_t i = 0; i < D; ++i) {
        equal = equal && (data_[i] == rhs[i]);
    }
    return equal;
}
//*/

template <typename T, size_t D>
struct CoordinateHashFunc {
    __device__ __host__ uint64_t operator()(const Coordinate<T, D>& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        /** We only support 4-byte and 8-byte types **/
        using input_t = typename std::conditional<sizeof(T) == sizeof(uint32_t),
                                                  uint32_t, uint64_t>::type;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            hash ^= *((input_t*)(&key[i]));
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

template class Coordinate<int, 4>;
template class Coordinate<int, 5>;
template class Coordinate<int, 6>;
template class Coordinate<int, 7>;
