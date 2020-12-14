/*
 * Copyright 2019 Saman Ashkiani,
 * Modified 2019 by Wei Dong
 *
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
 *
 */

#pragma once

#include <thrust/device_vector.h>
#include "slab_hash/slab_hash.h"

/*
 * Default hash function:
 * It treat any kind of input as a concatenation of ints.
 */
template <typename Key>
struct hash {
    __device__ __host__ uint64_t operator()(const Key& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = sizeof(Key) / sizeof(int);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= ((int32_t*)(&key))[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

/* Lightweight wrapper to handle host input */
/* Key supports elementary types: int, long, etc. */
/* Value supports arbitrary types in theory. */
/* std::vector<bool> is specialized: it stores only one bit per element
 * We have to use uint8_t instead to read and write masks
 * https://en.wikipedia.org/w/index.php?title=Sequence_container_(C%2B%2B)&oldid=767869909#Specialization_for_bool
 */
namespace cuda {
using slab_hash::SlabHash;
using slab_hash::CudaAllocator;

template <typename Key,
          typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS = 5,
          uint32_t LOG_NUM_MEM_BLOCKS = 5,
          typename Hash = hash<Key>,
          class Alloc = CudaAllocator>
class unordered_map {
public:
    using key_type = Key;
    using value_type = Value;
public:
//    static constexpr uint32_t LOG_NUM_MEM_BLOCKS = 5;
//    static constexpr uint32_t LOG_NUM_SUPER_BLOCKS = 5;
    static constexpr uint32_t key_chunks = sizeof(Key) / sizeof(uint32_t);
    static constexpr uint32_t value_chunks = sizeof(Value) / sizeof(uint32_t);
    static constexpr uint32_t MEM_UNIT_WARP_MULTIPLES = key_chunks + value_chunks;
//      (sizeof(Key) + sizeof(Value)) / sizeof(uint32_t);
public:
    unordered_map() {}
    unordered_map(uint32_t max_keys,
                  /* Preset hash table params to estimate bucket num */
                  float duplicate_factor =
                      1.0 / pow(2, sizeof(Key) / sizeof(uint32_t) - 1),
                  uint32_t keys_per_bucket = 31 * 2,
                  /* CUDA device */
                  const uint32_t device_idx = 0);
    ~unordered_map();

    void reserve(uint32_t max_keys,
                  /* Preset hash table params to estimate bucket num */
                  float duplicate_factor =
                      1.0 / pow(2, sizeof(Key) / sizeof(uint32_t) - 1),
                  uint32_t keys_per_bucket = 31 * 2,
                  /* CUDA device */
                  const uint32_t device_idx = 0);

    /* Minimal output */
    /* No output for Insert */
    Value Size();
    Value BulkBuild(const std::vector<Key>& input_keys);
    Value BulkBuild(thrust::device_vector<Key>& input_keys);
    Value BulkBuild(Key* input_keys, int num_keys);

    /* Value and mask output for Search */
    std::pair<thrust::device_vector<Value>, thrust::device_vector<uint8_t>>
    Search(const std::vector<Key>& input_keys);
    std::pair<thrust::device_vector<Value>, thrust::device_vector<uint8_t>>
    Search(thrust::device_vector<Key>& input_keys);
    std::pair<thrust::device_vector<Value>, thrust::device_vector<uint8_t>>
    Search(Key* input_keys, int num_keys);

    /* No output for Remove */
    void Remove(const std::vector<Key>& input_keys);
    void Remove(thrust::device_vector<Key>& input_keys);
    void Remove(Key* input_keys, int num_keys);

    std::pair<thrust::device_vector<_Iterator<Key, Value>>,
              thrust::device_vector<uint8_t>>
    Search_(thrust::device_vector<Key>& input_keys);

    thrust::device_vector<uint8_t> Remove_(
            thrust::device_vector<Key>& input_keys);

    /* Assistance functions */
    float ComputeLoadFactor();
    std::vector<int> CountElemsPerBucket();
    void CountElems(thrust::device_vector<int>& count);

    //////////////
    void BulkInsert(const int* p_coords,
                       int* p_mapping,
                       int* p_inverse_mapping,
                       int size, int key_chunks_);
    void IterateKeys(int* p_coords, int size);
    void IterateSearchAtBatch(int* p_out, int batch_index, int size);
    void IterateSearchPerBatch(const std::vector<int*>& p_outs, int size);
    void IterateOffsetInsert(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                             int* p_offset, int size);
    void IterateOffsetInsertWithInsOuts(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        int* p_offset,
                                        int* p_in, int* p_out,
                                        int size);
    void IterateOffsetSearch(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        int* p_offset,
                                        int* p_in, int* p_out,
                                        int size);
    void IterateBatchInsert(const std::shared_ptr<unordered_map<Key, Value,
                                                LOG_NUM_SUPER_BLOCKS,
                                                LOG_NUM_MEM_BLOCKS,
                                                Hash, Alloc>>& in_map,
                                        int size);
    void IterateBatchSearch(const std::shared_ptr<unordered_map<Key, Value,
                                                LOG_NUM_SUPER_BLOCKS,
                                                LOG_NUM_MEM_BLOCKS,
                                                Hash, Alloc>>& in_map,
                                        int* p_in, int* p_out,
                                        int size);
    void IterateStrideInsert(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        const std::vector<int>& tensor_strides,
                                        int size);
    void IterateStrideInsertWithInOut(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        int* p_in, int* p_out,
                                        const std::vector<int>& tensor_strides,
                                        int size);
    void IterateStrideSearch(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        int* p_in, int* p_out,
                                        const std::vector<int>& tensor_strides,
                                        int size);
    void IterateInsert(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                                 int size);
    void IterateInsertWithInsOuts(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        int* p_in, int* p_out,
                                                 int size);
    void IterateSearch(const std::shared_ptr<unordered_map<Key, Value,
                                                 LOG_NUM_SUPER_BLOCKS,
                                                 LOG_NUM_MEM_BLOCKS,
                                                 Hash, Alloc>>& in_map,
                                        int* p_in, int* p_out,
                                                 int size);
    void IteratePruneInsert(const std::shared_ptr<unordered_map<Key, Value,
                                                      LOG_NUM_SUPER_BLOCKS,
                                                      LOG_NUM_MEM_BLOCKS,
                                                      Hash, Alloc>>& in_map,
                                    bool* p_keep, int keep_size,
                                    int size);
    void IteratePruneInsertWithInOut(const std::shared_ptr<unordered_map<Key, Value,
                                                      LOG_NUM_SUPER_BLOCKS,
                                                      LOG_NUM_MEM_BLOCKS,
                                                      Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                    bool* p_keep, int keep_size,
                                    int size);
    void IteratePruneSearch(const std::shared_ptr<unordered_map<Key, Value,
                                                      LOG_NUM_SUPER_BLOCKS,
                                                      LOG_NUM_MEM_BLOCKS,
                                                      Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                    bool* p_keep, int keep_size,
                                    int size);
    //////////////

    const std::shared_ptr<SlabHash<Key, Value, Hash, Alloc,
                             LOG_NUM_MEM_BLOCKS,
                             LOG_NUM_SUPER_BLOCKS,
                             MEM_UNIT_WARP_MULTIPLES>>&
    get_slab_hash() const {
      return slab_hash_;
    }

private:
    uint32_t max_keys_;
    uint32_t num_buckets_;
    uint32_t cuda_device_idx_;

    /* Buffer for input cpu data (e.g. from std::vector) */
    Key* input_key_buffer_;
    Key* output_key_buffer_;
    Value* output_value_buffer_;
    _Iterator<Key, Value>* output_iterator_buffer_;
    uint8_t* output_mask_buffer_;

    std::shared_ptr<SlabHash<Key, Value, Hash, Alloc,
                             LOG_NUM_MEM_BLOCKS,
                             LOG_NUM_SUPER_BLOCKS,
                             MEM_UNIT_WARP_MULTIPLES>> slab_hash_;
    std::shared_ptr<Alloc> allocator_;
};

/*
template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::unordered_map(
        uint32_t max_keys,
        float duplicate_factor,
        uint32_t keys_per_bucket,
        const uint32_t device_idx) {
    reserve(max_keys, duplicate_factor, keys_per_bucket, device_idx);
}
*/

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::unordered_map(
        uint32_t max_keys,
        float duplicate_factor,
        uint32_t keys_per_bucket,
        const uint32_t device_idx) {
    max_keys_ = max_keys;
    cuda_device_idx_ = device_idx;
    /* Set bucket size */
    uint32_t expected_unique_keys = max_keys * duplicate_factor;
    num_buckets_ = (expected_unique_keys + keys_per_bucket - 1) / keys_per_bucket;

    /* Set device */
    int32_t cuda_device_count_ = 0;
    CHECK_CUDA(cudaGetDeviceCount(&cuda_device_count_));
    assert(cuda_device_idx_ < cuda_device_count_);
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    allocator_ = std::make_shared<Alloc>(cuda_device_idx_);

    // allocating key, value arrays to buffer input and output:
    input_key_buffer_ = allocator_->template allocate<Key>(max_keys_);
    output_key_buffer_ = allocator_->template allocate<Key>(max_keys_);
    output_value_buffer_ = allocator_->template allocate<Value>(max_keys_);
    output_mask_buffer_ = allocator_->template allocate<uint8_t>(max_keys_);
    output_iterator_buffer_ =
            allocator_->template allocate<_Iterator<Key, Value>>(max_keys_);

    // allocate an initialize the allocator:
    slab_hash_ = std::make_shared<SlabHash<Key, Value, Hash, Alloc,
                                           LOG_NUM_MEM_BLOCKS,
                                           LOG_NUM_SUPER_BLOCKS,
                                           MEM_UNIT_WARP_MULTIPLES>>(
            num_buckets_, max_keys_, cuda_device_idx_);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::~unordered_map() {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    allocator_->template deallocate<Key>(input_key_buffer_);
    allocator_->template deallocate<Key>(output_key_buffer_);
    allocator_->template deallocate<Value>(output_value_buffer_);
    allocator_->template deallocate<uint8_t>(output_mask_buffer_);
    allocator_->template deallocate<_Iterator<Key, Value>>(
            output_iterator_buffer_);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
Value unordered_map<Key, Value,
                    LOG_NUM_SUPER_BLOCKS,
                    LOG_NUM_MEM_BLOCKS,
                    Hash, Alloc>::Size() {
    auto elems_per_bucket = slab_hash_->CountElemsPerBucket();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);
    return total_elems_stored;
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
Value unordered_map<Key, Value,
                    LOG_NUM_SUPER_BLOCKS,
                    LOG_NUM_MEM_BLOCKS,
                    Hash, Alloc>::BulkBuild(
        const std::vector<Key>& input_keys) {
    assert(input_keys.size() <= max_keys_);

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(input_key_buffer_, input_keys.data(),
                          sizeof(Key) * input_keys.size(),
                          cudaMemcpyHostToDevice));

    slab_hash_->InsertAtomic(input_key_buffer_,
                       input_keys.size());
    return slab_hash_->Size();
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
Value unordered_map<Key, Value,
                    LOG_NUM_SUPER_BLOCKS,
                    LOG_NUM_MEM_BLOCKS,
                    Hash, Alloc>::BulkBuild(
        thrust::device_vector<Key>& input_keys) {
    assert(input_keys.size() <= max_keys_);

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    slab_hash_->InsertAtomic(thrust::raw_pointer_cast(input_keys.data()),
                             input_keys.size());
    return slab_hash_->Size();
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
Value unordered_map<Key, Value,
                    LOG_NUM_SUPER_BLOCKS,
                    LOG_NUM_MEM_BLOCKS,
                    Hash, Alloc>::BulkBuild(Key* input_keys,
                                                    int num_keys) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    slab_hash_->InsertAtomic(input_keys, num_keys);
    return slab_hash_->Size();
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
std::pair<thrust::device_vector<Value>, thrust::device_vector<uint8_t>>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::Search(
        const std::vector<Key>& input_keys) {
    assert(input_keys.size() <= max_keys_);

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemset(output_mask_buffer_, 0,
                          sizeof(uint8_t) * input_keys.size()));

    CHECK_CUDA(cudaMemcpy(input_key_buffer_, input_keys.data(),
                          sizeof(Key) * input_keys.size(),

                          cudaMemcpyHostToDevice));
    slab_hash_->Search(input_key_buffer_, output_value_buffer_,
                       output_mask_buffer_, input_keys.size());
    CHECK_CUDA(cudaDeviceSynchronize());

    thrust::device_vector<Value> output_values(
            output_value_buffer_, output_value_buffer_ + input_keys.size());
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys.size());
    return std::make_pair(output_values, output_masks);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
std::pair<thrust::device_vector<Value>, thrust::device_vector<uint8_t>>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::Search(
        thrust::device_vector<Key>& input_keys) {
    assert(input_keys.size() <= max_keys_);
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemset(output_mask_buffer_, 0,
                          sizeof(uint8_t) * input_keys.size()));

    slab_hash_->Search(thrust::raw_pointer_cast(input_keys.data()),
                       output_value_buffer_, output_mask_buffer_,
                       input_keys.size());
    CHECK_CUDA(cudaDeviceSynchronize());

    thrust::device_vector<Value> output_values(
            output_value_buffer_, output_value_buffer_ + input_keys.size());
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys.size());
    return std::make_pair(output_values, output_masks);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
std::pair<thrust::device_vector<Value>, thrust::device_vector<uint8_t>>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::Search(Key* input_keys, int num_keys) {
    assert(num_keys <= max_keys_);
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemset(output_mask_buffer_, 0, sizeof(uint8_t) * num_keys));

    slab_hash_->Search(input_keys, output_value_buffer_, output_mask_buffer_,
                       num_keys);
    CHECK_CUDA(cudaDeviceSynchronize());

    thrust::device_vector<Value> output_values(output_value_buffer_,
                                               output_value_buffer_ + num_keys);
    thrust::device_vector<uint8_t> output_masks(output_mask_buffer_,
                                                output_mask_buffer_ + num_keys);
    return std::make_pair(output_values, output_masks);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void unordered_map<Key, Value,
                   LOG_NUM_SUPER_BLOCKS,
                   LOG_NUM_MEM_BLOCKS,
                   Hash, Alloc>::Remove(
        const std::vector<Key>& input_keys) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(input_key_buffer_, input_keys.data(),
                          sizeof(Key) * input_keys.size(),
                          cudaMemcpyHostToDevice));
    slab_hash_->Remove(input_key_buffer_, input_keys.size());
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void unordered_map<Key, Value,
                   LOG_NUM_SUPER_BLOCKS,
                   LOG_NUM_MEM_BLOCKS,
                   Hash, Alloc>::Remove(
        thrust::device_vector<Key>& input_keys) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    slab_hash_->Remove(thrust::raw_pointer_cast(input_keys.data()),
                       input_keys.size());
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void unordered_map<Key, Value,
                   LOG_NUM_SUPER_BLOCKS,
                   LOG_NUM_MEM_BLOCKS,
                   Hash, Alloc>::Remove(Key* input_keys,
                                                    int num_keys) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    slab_hash_->Remove(input_keys, num_keys);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
std::pair<thrust::device_vector<_Iterator<Key, Value>>,
          thrust::device_vector<uint8_t>>
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::Search_(
        thrust::device_vector<Key>& input_keys) {
    assert(input_keys.size() <= max_keys_);

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemset(output_mask_buffer_, 0,
                          sizeof(uint8_t) * input_keys.size()));

    slab_hash_->Search_(thrust::raw_pointer_cast(input_keys.data()),
                        output_iterator_buffer_, output_mask_buffer_,
                        input_keys.size());
    CHECK_CUDA(cudaDeviceSynchronize());

    thrust::device_vector<_Iterator<Key, Value>> output_iterators(
            output_iterator_buffer_,
            output_iterator_buffer_ + input_keys.size());
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys.size());
    return std::make_pair(output_iterators, output_masks);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
std::vector<int> unordered_map<Key, Value,
                               LOG_NUM_SUPER_BLOCKS,
                               LOG_NUM_MEM_BLOCKS,
                               Hash, Alloc>::CountElemsPerBucket() {
    return slab_hash_->CountElemsPerBucket();
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void unordered_map<Key, Value,
                   LOG_NUM_SUPER_BLOCKS,
                   LOG_NUM_MEM_BLOCKS,
                   Hash, Alloc>::CountElems(
    thrust::device_vector<int>& count) {
    std::vector<int> std_count(3);
    thrust::copy(count.begin(), count.end(), std_count.begin());
    printf("Before count: %d\t%d\t%d\n", std_count[0], std_count[1], std_count[2]);
    slab_hash_->CountElems(thrust::raw_pointer_cast(count.data()));
    thrust::copy(count.begin(), count.end(), std_count.begin());
    printf("After count: %d\t%d\t%d\n", std_count[0], std_count[1], std_count[2]);
    assert(std_count[0] == 0);
    assert(std_count[1] == 0);
    assert(std_count[2] == 0);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
float unordered_map<Key, Value,
                    LOG_NUM_SUPER_BLOCKS,
                    LOG_NUM_MEM_BLOCKS,
                    Hash, Alloc>::ComputeLoadFactor() {
    return slab_hash_->ComputeLoadFactor();
}
}  // namespace cuda
