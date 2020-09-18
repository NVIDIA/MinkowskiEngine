/*
 * Copyright 2019 Saman Ashkiani
 * Modified 2019 by Wei Dong
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cassert>
#include <memory>
#include <random>
#include <time.h>

#include "slab_alloc.h"

template <typename _Key, typename _Value>
struct _Pair {
    _Key first;
    _Value second;
    __device__ __host__ _Pair(const _Key& key, const _Value& value)
        : first(key), second(value) {}
    __device__ __host__ _Pair() : first(), second() {}
};

template <typename _Key, typename _Value>
__device__ __host__ _Pair<_Key, _Value> _make_pair(const _Key& key,
                                                   const _Value& value) {
    return _Pair<_Key, _Value>(key, value);
}

template <typename _Key, typename _Value>
using _Iterator = _Pair<_Key, _Value>*;

namespace slab_hash {

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
class SlabHashContext;

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
class SlabHash {
public:
    SlabHash(const uint32_t max_bucket_count,
             const uint32_t max_keyvalue_count,
             uint32_t device_idx);

    ~SlabHash();

    _Value Size();
    _Value* SizePtr();
    /* Simplistic output: no iterators, and success mask is only provided
     * for search.
     * All the outputs are READ ONLY: change to these output will NOT change the
     * internal hash table.
     */
    void InsertAtomic(_Key* input_keys, uint32_t num_keys);
    void Search(_Key* input_keys,
                _Value* output_values,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Remove(_Key* input_keys, uint32_t num_keys);

    /* Verbose output (similar to std): return success masks for all operations,

    * and iterators for insert and search (not for remove operation, as
     * iterators are invalid after erase).
     * Output iterators supports READ/WRITE: change to these output will
     * DIRECTLY change the internal hash table.
     */
    void InsertAtomic_(_Key* input_keys,
                 _Iterator<_Key, _Value>* output_iterators,
                 uint8_t* output_masks,
                 uint32_t num_keys);
    void Search_(_Key* input_keys,
                 _Iterator<_Key, _Value>* output_iterators,
                 uint8_t* output_masks,
                 uint32_t num_keys);
    void Remove_(_Key* input_keys, uint8_t* output_masks, uint32_t num_keys);

    ////////////////
    void BulkInsertWithMapping(const int* p_coords,
                   int* p_mapping,
                   int* p_inverse_mapping,
                   int size);
    void IterateKeys(int* p_coords, int size);
    void IterateSearchAtBatch(int* p_out, int batch_index, int size);
    void IterateSearchPerBatch(const std::vector<int*>& p_outs, int size);
    void IterateOffsetInsert(const std::shared_ptr<SlabHash<_Key, _Value,
                                             _Hash, _Alloc,
                                             _LOG_NUM_MEM_BLOCKS,
                                             _LOG_NUM_SUPER_BLOCKS,
                                             _MEM_UNIT_WARP_MULTIPLES>>& in_map,
                         int* p_offset, int size);
    ////////////////

    /* Debug usages */
    std::vector<int> CountElemsPerBucket();

    void CountElems(int* count);

    double ComputeLoadFactor();

private:
    ptr_t* bucket_list_head_;
    uint32_t num_buckets_;

    _Value* cnt_value_;

    SlabHashContext<_Key, _Value, _Hash,
                    _LOG_NUM_MEM_BLOCKS,
                    _LOG_NUM_SUPER_BLOCKS,
                    _MEM_UNIT_WARP_MULTIPLES> gpu_context_;

    std::shared_ptr<_Alloc> allocator_;
    SlabAlloc<_Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>* slab_list_allocator_;

    uint32_t device_idx_;

    uint32_t hash_coef_;
};

/** Lite version **/
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void InsertAtomicKernel(SlabHashContext<_Key, _Value, _Hash,
                                                   _LOG_NUM_MEM_BLOCKS,
                                                   _LOG_NUM_SUPER_BLOCKS,
                                                   _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
                             _Key* input_keys,
                             uint32_t num_keys,
                             uint32_t hash_coef);
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void SearchKernel(SlabHashContext<_Key, _Value, _Hash,
                                             _LOG_NUM_MEM_BLOCKS,
                                             _LOG_NUM_SUPER_BLOCKS,
                                             _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
                             _Key* input_keys,
                             _Value* output_values,
                             uint8_t* output_masks,
                             uint32_t num_keys);
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void RemoveKernel(SlabHashContext<_Key, _Value, _Hash,
                                             _LOG_NUM_MEM_BLOCKS,
                                             _LOG_NUM_SUPER_BLOCKS,
                                             _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
                             _Key* input_keys,
                             uint32_t num_keys);

/** Verbose version **/
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void InsertAtomic_Kernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        _Key* input_keys,
        _Iterator<_Key, _Value>* output_iterators,
        uint8_t* output_masks,
        uint32_t num_keys,
        uint32_t hash_coef);
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void Search_Kernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        _Key* input_keys,
        _Iterator<_Key, _Value>* output_iterators,
        uint8_t* output_masks,
        uint32_t num_keys);
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void Remove_Kernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        _Key* input_keys,
        uint8_t* output_masks,
        uint32_t num_keys);

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void GetIteratorsKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        _Iterator<_Key, _Value>* output_iterators,
        uint32_t* output_iterator_count,
        uint32_t num_buckets);
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void CountElemsPerBucketKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        uint32_t* bucket_elem_counts, uint32_t hash_coef);

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void CountElemsKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        uint32_t* values, uint32_t* index);

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void InitKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        const uint32_t num_threads,
        uint32_t hash_coef);

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void ReleaseKernel(const uint32_t num_threads);

///////////////
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void BulkInsertWithMappingKernel(SlabHashContext<_Key, _Value, _Hash,
                                                   _LOG_NUM_MEM_BLOCKS,
                                                   _LOG_NUM_SUPER_BLOCKS,
                                                   _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
//                             _Key* input_keys,
                             const int* p_coords,
                             int* p_mapping,
                             int* p_inverse_mapping,
                             uint32_t num_keys,
                             uint32_t hash_coef);
///////////////

/**
 * Internal implementation for the device proxy:
 * DO NOT ENTER!
 **/
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
class SlabHashContext {
public:
   using SlabAllocContextT = SlabAllocContext<_LOG_NUM_MEM_BLOCKS,
                                              _LOG_NUM_SUPER_BLOCKS,
                                              _MEM_UNIT_WARP_MULTIPLES>;
public:
    SlabHashContext();
    __host__ void Setup(
            ptr_t* bucket_list_head,
            const uint32_t num_buckets,
            _Value* cnt_value,
            const SlabAllocContextT& allocator_ctx) {
    bucket_list_head_ = bucket_list_head;
    num_buckets_ = num_buckets;
    cnt_value_ = cnt_value;
    slab_list_allocator_ctx_ = allocator_ctx;
    }

    /* Core SIMT operations, shared by both simplistic and verbose
     * interfaces */
    __device__ _Pair<ptr_t*, uint8_t> InsertAtomic(uint8_t& lane_active,
                                            const uint32_t lane_id,
                                            const uint32_t bucket_id,
                                            const _Key& key);
    __device__ _Pair<ptr_t*, uint8_t> Search(uint8_t& lane_active,
                                            const uint32_t lane_id,
                                            const uint32_t bucket_id,
                                            const _Key& key);

    __device__ uint8_t Remove(uint8_t& lane_active,
                              const uint32_t lane_id,
                              const uint32_t bucket_id,
                              const _Key& key);

    /////////////////
    __device__ void BulkInsertWithMapping(uint8_t& lane_active,
                                 const uint32_t lane_id,
                                 const uint32_t bucket_id,
                                 const _Key& key,
                                 int* p_mapping,
                                 int* p_inverse_mapping,
                                 int key_idx);
    /////////////////

    /* Hash function */
    __device__ __host__ uint32_t ComputeBucket(const _Key& key) const;
    __device__ __host__ uint32_t bucket_size() const { return num_buckets_; }

    __device__ __host__ SlabAllocContextT& get_slab_alloc_ctx() {
        return slab_list_allocator_ctx_;
    }

    __device__ __forceinline__ ptr_t* get_unit_ptr_from_list_nodes(
            const ptr_t slab_ptr, const uint32_t lane_id) {
        return slab_list_allocator_ctx_.get_unit_ptr_from_slab(slab_ptr,
                                                               lane_id);
    }
    __device__ __forceinline__ ptr_t* get_slab_ptr_from_list_head(
            const uint32_t bucket_id) {
        return bucket_list_head_ + bucket_id;
    }
    __device__ __forceinline__ void ClearRemainPair(ptr_t* unit_data_ptr);

private:
    __device__ __forceinline__ void CopyRemainPair(ptr_t* unit_data_ptr,
                                                   const _Key& key,
                                                   const _Value& value);
    __device__ __forceinline__ void WarpSyncKey(const _Key& key,
                                                const uint32_t lane_id,
                                                _Key& ret);
    __device__ __forceinline__ int32_t WarpFindKey(const _Key& src_key,
                                                   const uint32_t lane_id,
                                                   const ptr_t* unit_data_ptr);
    __device__ __forceinline__ int32_t WarpFindEmpty(const ptr_t* unit_data_ptr);

    __device__ __forceinline__ ptr_t AllocateSlab(const uint32_t lane_id);
    __device__ __forceinline__ void FreeSlab(const ptr_t slab_ptr);

private:
    uint32_t num_buckets_;
    _Hash hash_fn_;

    ptr_t* bucket_list_head_;
    _Value* cnt_value_;
    SlabAllocContext<_LOG_NUM_MEM_BLOCKS,
                     _LOG_NUM_SUPER_BLOCKS,
                     _MEM_UNIT_WARP_MULTIPLES> slab_list_allocator_ctx_;

public:
    static constexpr uint32_t key_chunks = sizeof(_Key) / sizeof(uint32_t);
    static constexpr uint32_t value_chunks = sizeof(_Value) / sizeof(uint32_t);
};


} // namespace slab_hash
