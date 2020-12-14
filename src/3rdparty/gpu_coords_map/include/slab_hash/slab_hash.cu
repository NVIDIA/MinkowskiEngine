#include <algorithm>
#include "slab_hash.h"
#include "../coordinate.h"
#include "../cuda_unordered_map.h"

namespace slab_hash {
/**
 * Implementation for the host class
 **/
template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
SlabHash<_Key, _Value, _Hash, _Alloc,
         _LOG_NUM_MEM_BLOCKS,
         _LOG_NUM_SUPER_BLOCKS,
         _MEM_UNIT_WARP_MULTIPLES>::SlabHash(
        const uint32_t max_bucket_count,
        const uint32_t max_keyvalue_count,
        uint32_t device_idx)
    : num_buckets_(max_bucket_count),
      device_idx_(device_idx),
      bucket_list_head_(nullptr) {
    int32_t device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    assert(device_idx_ < device_count);
    CHECK_CUDA(cudaSetDevice(device_idx_));

    // allocate an initialize the allocator:
    allocator_ = std::make_shared<_Alloc>(device_idx);
    slab_list_allocator_ = SlabAlloc<_Alloc,
                                     _LOG_NUM_MEM_BLOCKS,
                                     _LOG_NUM_SUPER_BLOCKS,
                                     _MEM_UNIT_WARP_MULTIPLES>::getInstance();

    assert(sizeof(_Value) % sizeof(ptr_t) == 0);
    // allocating initial buckets:
    bucket_list_head_ = allocator_->template allocate<ptr_t>(num_buckets_ +
        sizeof(_Value) / sizeof(ptr_t));
    cnt_value_ = reinterpret_cast<_Value*>(bucket_list_head_ + num_buckets_);
    CHECK_CUDA(
            cudaMemset(bucket_list_head_, 0xFF, sizeof(ptr_t) * num_buckets_));
    CHECK_CUDA(
            cudaMemset(cnt_value_, 0x00, sizeof(_Value)));

    gpu_context_.Setup(bucket_list_head_, num_buckets_,
                       cnt_value_,
                       slab_list_allocator_->getContext());

    // random coefficients for allocator's hash function
    std::mt19937 rng(time(0));
    hash_coef_ = rng();

    std::cout << "hash_coef_: " << hash_coef_ << std::endl;

    const uint32_t num_threads = num_buckets_ * WARP_WIDTH;
    const uint32_t num_blocks = (num_threads + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    InitKernel<_Key, _Value, _Hash><<<num_blocks, BLOCKSIZE_>>>(gpu_context_, num_threads, hash_coef_);
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
SlabHash<_Key, _Value, _Hash, _Alloc,
         _LOG_NUM_MEM_BLOCKS,
         _LOG_NUM_SUPER_BLOCKS,
         _MEM_UNIT_WARP_MULTIPLES>::~SlabHash() {
    CHECK_CUDA(cudaSetDevice(device_idx_));

    slab_list_allocator_->getContext() = gpu_context_.get_slab_alloc_ctx();
    auto slabs_per_super_block = slab_list_allocator_->CountSlabsPerSuperblock();
    int total_slabs_stored = std::accumulate(
            slabs_per_super_block.begin(), slabs_per_super_block.end(), 0);

    std::cout << "Before total_slabs_stored: " << total_slabs_stored << std::endl;

    for (auto n : slabs_per_super_block) std::cout << n << '\t'; std::cout << std::endl;

    auto elems_per_bucket = CountElemsPerBucket();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    printf("Before total_elems_stored: %d\n", total_elems_stored);

    const uint32_t num_threads = num_buckets_ * 32;
    const uint32_t num_blocks = (num_threads + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    ReleaseKernel<_Key, _Value, _Hash><<<num_blocks, BLOCKSIZE_>>>(gpu_context_, num_threads);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());

    elems_per_bucket = CountElemsPerBucket();
    total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    printf("After total_elems_stored: %d\n", total_elems_stored);

    slabs_per_super_block = slab_list_allocator_->CountSlabsPerSuperblock();
    total_slabs_stored = std::accumulate(
            slabs_per_super_block.begin(), slabs_per_super_block.end(), 0);

    std::cout << "After total_slabs_stored: " << total_slabs_stored << std::endl;

    for (auto n : slabs_per_super_block) {
        std::cout << n << '\t';
//        assert(n == 0);
    }
    std::cout << std::endl;

    allocator_->template deallocate(bucket_list_head_);
    std::cout << num_buckets_ << std::endl;
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
_Value SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::Size() {

    _Value cnt_value;

    CHECK_CUDA(cudaMemcpy(&cnt_value, cnt_value_,
                          sizeof(_Value),
                          cudaMemcpyDeviceToHost));

    return cnt_value;
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
_Value* SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::SizePtr() {
    return cnt_value_;
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::InsertAtomic(_Key* keys,
                                                      uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    CHECK_CUDA(cudaSetDevice(device_idx_));
    InsertAtomicKernel<_Key, _Value, _Hash,
                       _LOG_NUM_MEM_BLOCKS,
                       _LOG_NUM_SUPER_BLOCKS,
                       _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, num_keys, hash_coef_);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::Search(_Key* keys,
                                                _Value* values,
                                                uint8_t* founds,
                                                uint32_t num_keys) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    SearchKernel<_Key, _Value, _Hash,
                 _LOG_NUM_MEM_BLOCKS,
                 _LOG_NUM_SUPER_BLOCKS,
                 _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, values, founds, num_keys);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::Remove(_Key* keys,
                                                uint32_t num_keys) {
    std::cout << "Enter Remove" << std::endl;
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    RemoveKernel<_Key, _Value, _Hash,
                 _LOG_NUM_MEM_BLOCKS,
                 _LOG_NUM_SUPER_BLOCKS,
                 _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, num_keys);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::InsertAtomic_(
        _Key* keys,
        _Iterator<_Key, _Value>* iterators,
        uint8_t* masks,
        uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    CHECK_CUDA(cudaSetDevice(device_idx_));
    InsertAtomic_Kernel<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, iterators, masks, num_keys, hash_coef_);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::Search_(_Key* keys,
                                                 _Iterator<_Key, _Value>* iterators,
                                                 uint8_t* masks,
                                                 uint32_t num_keys) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    Search_Kernel<_Key, _Value, _Hash,
                  _LOG_NUM_MEM_BLOCKS,
                  _LOG_NUM_SUPER_BLOCKS,
                  _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, iterators, masks, num_keys);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::Remove_(_Key* keys,
                                                 uint8_t* masks,
                                                 uint32_t num_keys) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    Remove_Kernel<_Key, _Value, _Hash,
                  _LOG_NUM_MEM_BLOCKS,
                  _LOG_NUM_SUPER_BLOCKS,
                  _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, masks, num_keys);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

/* Debug usage */
template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
std::vector<int> SlabHash<_Key, _Value, _Hash, _Alloc,
                          _LOG_NUM_MEM_BLOCKS,
                          _LOG_NUM_SUPER_BLOCKS,
                          _MEM_UNIT_WARP_MULTIPLES>::CountElemsPerBucket() {
    std::cout << "num_buckets_: " << num_buckets_ << std::endl;

    auto elems_per_bucket_buffer =
            allocator_->template allocate<uint32_t>(num_buckets_);

    thrust::device_vector<uint32_t> elems_per_bucket(
            elems_per_bucket_buffer, elems_per_bucket_buffer + num_buckets_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const uint32_t blocksize = 128;
    const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
    CountElemsPerBucketKernel<_Key, _Value, _Hash,
                              _LOG_NUM_MEM_BLOCKS,
                              _LOG_NUM_SUPER_BLOCKS,
                              _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, blocksize>>>(
            gpu_context_, thrust::raw_pointer_cast(elems_per_bucket.data()), hash_coef_);

    std::vector<int> result(num_buckets_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    allocator_->template deallocate<uint32_t>(elems_per_bucket_buffer);
    return std::move(result);
}

/* Debug usage */
template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void SlabHash<_Key, _Value, _Hash, _Alloc,
              _LOG_NUM_MEM_BLOCKS,
              _LOG_NUM_SUPER_BLOCKS,
              _MEM_UNIT_WARP_MULTIPLES>::CountElems(int* count) {

    _Value cnt_value = Size();

    auto values_buffer =
            allocator_->template allocate<uint32_t>(cnt_value);

    auto index_buffer =
            allocator_->template allocate<uint32_t>(1);

    thrust::device_vector<uint32_t> values(
            values_buffer, values_buffer + cnt_value);

    thrust::device_vector<uint32_t> index(
            index_buffer, index_buffer + 1);
    thrust::fill(index.begin(), index.end(), 0);

    const uint32_t blocksize = 128;
    const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
    std::cout << "Before CountElemsKernel" << std::endl;
    CountElemsKernel<_Key, _Value, _Hash,
                     _LOG_NUM_MEM_BLOCKS,
                     _LOG_NUM_SUPER_BLOCKS,
                     _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, blocksize>>>(
                gpu_context_,
                thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(index.data()),
                count
            );
    std::cout << "After CountElemsKernel" << std::endl;

    std::vector<int> sorted_values(cnt_value);
    std::vector<int> cnt(1);
    thrust::copy(values.begin(), values.end(),
                 sorted_values.begin());
    thrust::copy(index.begin(), index.end(),
                 cnt.begin());
    allocator_->template deallocate<uint32_t>(values_buffer);
    allocator_->template deallocate<uint32_t>(index_buffer);

    std::cout << "Total Values: " << cnt[0] << std::endl;
    std::sort(sorted_values.begin(), sorted_values.begin() + cnt[0]);
    for (int i = 0; i != cnt[0]; ++i) {
        if (i != sorted_values[i]) std::cout << i << '\t' << sorted_values[i] << std::endl;
        assert(i == sorted_values[i]);
    }
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
double SlabHash<_Key, _Value, _Hash, _Alloc,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::ComputeLoadFactor() {
    auto elems_per_bucket = CountElemsPerBucket();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    slab_list_allocator_->getContext() = gpu_context_.get_slab_alloc_ctx();
    auto slabs_per_super_block = slab_list_allocator_->CountSlabsPerSuperblock();
    int total_slabs_stored = std::accumulate(
            slabs_per_super_block.begin(), slabs_per_super_block.end(), num_buckets_);

    double load_factor = double(total_elems_stored) /
                         double(total_slabs_stored * WARP_WIDTH);

    return load_factor;
}

////////////////
template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void
SlabHash<_Key, _Value, _Hash, _Alloc,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::
BulkInsertWithMapping(const int* p_coords,
               int* p_mapping,
               int* p_inverse_mapping,
               int num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    CHECK_CUDA(cudaSetDevice(device_idx_));
    BulkInsertWithMappingKernel<_Key, _Value, _Hash,
                       _LOG_NUM_MEM_BLOCKS,
                       _LOG_NUM_SUPER_BLOCKS,
                       _MEM_UNIT_WARP_MULTIPLES>
            <<<num_blocks, BLOCKSIZE_>>>(gpu_context_, p_coords,
                                         p_mapping, p_inverse_mapping,
                                         num_keys, hash_coef_);
//    CHECK_CUDA(cudaDeviceSynchronize());
//    CHECK_CUDA(cudaGetLastError());
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void
SlabHash<_Key, _Value, _Hash, _Alloc,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::
IterateKeys(int* p_coords, int size) {
}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void
SlabHash<_Key, _Value, _Hash, _Alloc,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::
IterateSearchAtBatch(int* p_out, int batch_index, int size) {}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void
SlabHash<_Key, _Value, _Hash, _Alloc,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::
IterateSearchPerBatch(const std::vector<int*>& p_outs, int size) {}

template <typename _Key, typename _Value, typename _Hash, class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
void
SlabHash<_Key, _Value, _Hash, _Alloc,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::
IterateOffsetInsert(const std::shared_ptr<SlabHash<_Key, _Value,
                                         _Hash, _Alloc,
                                         _LOG_NUM_MEM_BLOCKS,
                                         _LOG_NUM_SUPER_BLOCKS,
                                         _MEM_UNIT_WARP_MULTIPLES>>& in_map,
                     int* p_offset, int size) {}
////////////////

/**
 * Definitions
 **/
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::SlabHashContext()
    : num_buckets_(0), bucket_list_head_(nullptr) {
}

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
    uint32_t hash_coef) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_threads) return;
    uint32_t lane_id = tid & 0x1F;
    uint32_t bucket_id = (tid >> 5);


    slab_hash_ctx.get_slab_alloc_ctx().Init(hash_coef, tid, lane_id);

    ptr_t new_next_slab_ptr = slab_hash_ctx.get_slab_alloc_ctx().WarpAllocate(lane_id);

    if (lane_id == NEXT_SLAB_PTR_LANE) {

        const uint32_t* unit_data_ptr =
                slab_hash_ctx.get_slab_ptr_from_list_head(bucket_id);

// TODO(ljm): Ideal should be OK.
            //Ideal:
//            *((unsigned int*)unit_data_ptr) = new_next_slab_ptr;

        ptr_t old_next_slab_ptr =
                atomicCAS((unsigned int*)unit_data_ptr,
                          EMPTY_SLAB_PTR, new_next_slab_ptr);

        if (old_next_slab_ptr != EMPTY_SLAB_PTR) {
            slab_hash_ctx.get_slab_alloc_ctx().FreeUntouched(new_next_slab_ptr);
        }
    }

}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void ReleaseKernel(
    SlabHashContext<_Key, _Value, _Hash,
                    _LOG_NUM_MEM_BLOCKS,
                    _LOG_NUM_SUPER_BLOCKS,
                    _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
    const uint32_t num_threads) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_threads) return;
    uint32_t lane_id = (tid & 0x1F);
    uint32_t bucket_id = (tid >> 5);

    ptr_t* curr_slab_ptr_ptr = NULL;
    ptr_t curr_slab_ptr = EMPTY_SLAB_PTR;
    if (lane_id == NEXT_SLAB_PTR_LANE) {
        curr_slab_ptr_ptr =
            slab_hash_ctx.get_slab_ptr_from_list_head(bucket_id);
        curr_slab_ptr = *(curr_slab_ptr_ptr);
    }
    curr_slab_ptr =
            __shfl_sync(ACTIVE_LANES_MASK, curr_slab_ptr,
                      NEXT_SLAB_PTR_LANE, WARP_WIDTH);
    while (curr_slab_ptr != EMPTY_SLAB_PTR) {

        ptr_t* unit_data_ptr =
            slab_hash_ctx.get_unit_ptr_from_list_nodes(curr_slab_ptr, lane_id);

        if (lane_id != NEXT_SLAB_PTR_LANE) {
            ptr_t old_first_key = atomicExch(unit_data_ptr, EMPTY_PAIR_PTR);
            if (old_first_key != EMPTY_PAIR_PTR) {
                slab_hash_ctx.ClearRemainPair(unit_data_ptr);
            }
        } else {
            // set empty first
            *(curr_slab_ptr_ptr) = EMPTY_SLAB_PTR;
            // no need atomicExch ideally
    //        atomicExch(curr_slab_ptr_ptr, EMPTY_PAIR_PTR);
            // then, free untouched
            slab_hash_ctx.get_slab_alloc_ctx().FreeUntouched(curr_slab_ptr);
            curr_slab_ptr_ptr = unit_data_ptr;
            curr_slab_ptr = *(curr_slab_ptr_ptr);
        }
        curr_slab_ptr =
            __shfl_sync(ACTIVE_LANES_MASK, curr_slab_ptr,
                      NEXT_SLAB_PTR_LANE, WARP_WIDTH);
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __host__ __forceinline__ uint32_t
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::ComputeBucket(
                    const _Key& key) const {
    return hash_fn_(key) % num_buckets_;
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __forceinline__ void
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::WarpSyncKey(
                    const _Key& key,
                    const uint32_t lane_id,
                    _Key& ret) {
#pragma unroll 1
    for (size_t i = 0; i != key_chunks; ++i) {
        ((int*)(&ret))[i] = __shfl_sync(ACTIVE_LANES_MASK, ((int*)(&key))[i],
                                        lane_id, WARP_WIDTH);
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __forceinline__ void
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::ClearRemainPair(ptr_t* ptr) {
#pragma unroll 1
    for (size_t i = 1;
                i != key_chunks + value_chunks;
                ++i) {
        ptr[i] = EMPTY_SLAB_PTR;
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __forceinline__ void
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::CopyRemainPair(
                    ptr_t* ptr,
                    const _Key& key,
                    const _Value& value) {
#pragma unroll 1
    for (size_t i = 1; i < key_chunks; ++i) {
        ((int*)(ptr))[i] = ((int*)(&key))[i];
    }
#pragma unroll 1
    for (size_t i = 0; i < value_chunks; ++i) {
        ((int*)(ptr))[key_chunks + i] = ((int*)(&value))[i];
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ int32_t
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::WarpFindKey(
        const _Key& key, const uint32_t lane_id, const ptr_t* ptr) {
    uint8_t is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & PAIR_PTR_LANES_MASK)
            && (reinterpret_cast<const _Pair<_Key, _Value>*>(ptr))->first == key;

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_found)) - 1;
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __forceinline__ int32_t
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::WarpFindEmpty(const ptr_t* ptr) {
    //assert(value_chunks == 1);
    uint8_t is_lane_empty = (reinterpret_cast<const uint32_t*>(ptr)[_MEM_UNIT_WARP_MULTIPLES - 1] == EMPTY_PAIR_PTR);
    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_empty)) - 1;
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __forceinline__ ptr_t
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::AllocateSlab(
                    const uint32_t lane_id) {
    return slab_list_allocator_ctx_.WarpAllocate(lane_id);
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ __forceinline__ void
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::FreeSlab(
        const ptr_t slab_ptr) {
    slab_list_allocator_ctx_.FreeUntouched(slab_ptr);
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ _Pair<ptr_t*, uint8_t>
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::Search(
        uint8_t& to_search,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const _Key& query_key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t curr_slab_ptr = EMPTY_SLAB_PTR;

    ptr_t* iterator = NULL;
    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_search))) {
        /** 0. Restart from linked list head if the last query is finished
         * **/
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        _Key src_key;
        WarpSyncKey(query_key, src_lane, src_key);

        curr_slab_ptr =
                (prev_work_queue != work_queue)
                        ? *(get_slab_ptr_from_list_head(src_bucket))
                        : curr_slab_ptr;

        /* Each lane in the warp reads a uint in the slab in parallel */
        const uint32_t* unit_data_ptr =
                        get_unit_ptr_from_list_nodes(curr_slab_ptr, lane_id);

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data_ptr);

        /** 1. Found in this slab, SUCCEED **/
        if (lane_found >= 0) {
            /* broadcast found value */
            uint64_t found_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, reinterpret_cast<uint64_t>(unit_data_ptr), lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                to_search = false;

                iterator = reinterpret_cast<ptr_t*>(found_pair_internal_ptr);
                mask = true;
            }
        }

        /** 2. Not found in this slab **/
        else {
            ptr_t unit_data = *(reinterpret_cast<const ptr_t*>(unit_data_ptr));
            /* broadcast next slab: lane 31 reads 'next' */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** 2.1. Next slab is empty, ABORT **/
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                if (lane_id == src_lane) {
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists, RESTART **/
            else {
                curr_slab_ptr = next_slab_ptr;
            }
        }

        prev_work_queue = work_queue;
    }

    return _make_pair(iterator, mask);
}

/*
 * Insert: ABORT if found
 * replacePair: REPLACE if found
 * WE DO NOT ALLOW DUPLICATE KEYS
 */
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ _Pair<ptr_t*, uint8_t>
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::InsertAtomic(
        uint8_t& to_be_inserted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const _Key& key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = EMPTY_SLAB_PTR;

    ptr_t* iterator = NULL;
    uint8_t mask = false;

    const uint32_t first_key = *reinterpret_cast<const uint32_t*>(&key);

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        curr_slab_ptr =
                (prev_work_queue != work_queue)
                        ? *(get_slab_ptr_from_list_head(src_bucket))
                        : curr_slab_ptr;

        /* Each lane in the warp reads a uint in the slab */
        uint32_t* unit_data_ptr =
                  get_unit_ptr_from_list_nodes(curr_slab_ptr, lane_id);

        int32_t lane_empty = WarpFindEmpty(unit_data_ptr);
        _Key src_key;
        WarpSyncKey(key, src_lane, src_key);
        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data_ptr);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            uint64_t lane_empty_data_ptr = __shfl_sync(ACTIVE_LANES_MASK,
                reinterpret_cast<uint64_t>(unit_data_ptr), lane_empty, WARP_WIDTH);
            unit_data_ptr = reinterpret_cast<uint32_t*>(lane_empty_data_ptr);
            if (lane_id == src_lane) {

                uint32_t old_first_data =
                        atomicCAS(unit_data_ptr, EMPTY_PAIR_PTR, first_key);

                /** Branch 2.1: SUCCEED **/
                if (old_first_data == EMPTY_PAIR_PTR) {
                    // copy the remaining data
                    _Value value = atomicAdd(cnt_value_, 1);

                    CopyRemainPair(unit_data_ptr, key, value);
                    to_be_inserted = false;

                    iterator = unit_data_ptr;
                    mask = true;
                }

                /** Branch 2.2: failed: RESTART
                 *  In the consequent attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a different key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, *reinterpret_cast<const ptr_t*>(unit_data_ptr),
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (next_slab_ptr != EMPTY_SLAB_PTR) {
                curr_slab_ptr = next_slab_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                ptr_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_PTR_LANE) {
                    const uint32_t* unit_data_ptr =
                            get_unit_ptr_from_list_nodes(
                                              curr_slab_ptr,
                                              NEXT_SLAB_PTR_LANE);

                    ptr_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)unit_data_ptr,
                                      EMPTY_SLAB_PTR, new_next_slab_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_PTR) {
                        FreeSlab(new_next_slab_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }

    return _make_pair(iterator, mask);
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ uint8_t
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::Remove(
                                             uint8_t& to_be_deleted,
                                             const uint32_t lane_id,
                                             const uint32_t bucket_id,
                                             const _Key& key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = EMPTY_SLAB_PTR;

    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_deleted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        _Key src_key;
        WarpSyncKey(key, src_lane, src_key);

        curr_slab_ptr =
                (prev_work_queue != work_queue)
                        ? *(get_slab_ptr_from_list_head(src_bucket))
                        : curr_slab_ptr;

        const uint32_t* unit_data_ptr =
                get_unit_ptr_from_list_nodes(curr_slab_ptr, lane_id);

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data_ptr);

        /** Branch 1: key found **/
        if (lane_found >= 0) {

            if (lane_id == src_lane) {
                uint32_t* unit_data_ptr =
                        get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                           lane_found);
                ptr_t pair_to_delete = *reinterpret_cast<uint32_t*>(&src_key);

                // TODO: keep in mind the potential double free problem
                ptr_t old_key_value_pair =
                        atomicCAS((unsigned int*)(unit_data_ptr),
                                  pair_to_delete, EMPTY_PAIR_PTR);

                /** Branch 1.1: this thread reset, free src_addr **/
                if (old_key_value_pair == pair_to_delete) {
                    ClearRemainPair(unit_data_ptr);
                    mask = true;
                }
                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
                to_be_deleted = false;
            }
        } else {  // no matching slot found:
            ptr_t unit_data = *(reinterpret_cast<const ptr_t*>(unit_data_ptr));
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                // not found:
                if (lane_id == src_lane) {
                    to_be_deleted = false;
                }
            } else {
                curr_slab_ptr = next_slab_ptr;
            }
        }
        prev_work_queue = work_queue;
    }

    return mask;
}

/////////////////
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__device__ void
SlabHashContext<_Key, _Value, _Hash,
                _LOG_NUM_MEM_BLOCKS,
                _LOG_NUM_SUPER_BLOCKS,
                _MEM_UNIT_WARP_MULTIPLES>::BulkInsertWithMapping(
        uint8_t& to_be_inserted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const _Key& key,
        int* p_mapping,
        int* p_inverse_mapping,
        int key_idx) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = EMPTY_SLAB_PTR;

    //ptr_t* iterator = NULL;
    //uint8_t mask = false;

    const uint32_t first_key = *reinterpret_cast<const uint32_t*>(&key);

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        curr_slab_ptr =
                (prev_work_queue != work_queue)
                        ? *(get_slab_ptr_from_list_head(src_bucket))
                        : curr_slab_ptr;

        /* Each lane in the warp reads a uint in the slab */
        uint32_t* unit_data_ptr =
                  get_unit_ptr_from_list_nodes(curr_slab_ptr, lane_id);

        int32_t lane_empty = WarpFindEmpty(unit_data_ptr);
        _Key src_key;
        WarpSyncKey(key, src_lane, src_key);
        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data_ptr);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            uint64_t found_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, reinterpret_cast<uint64_t>(unit_data_ptr), lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                ///
                p_inverse_mapping[key_idx] = static_cast<int>(
                    *reinterpret_cast<_Value*>(
                        reinterpret_cast<ptr_t*>(
                          found_pair_internal_ptr)
                        + key_chunks)
                    );
                ///
                to_be_inserted = false;
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            uint64_t lane_empty_data_ptr = __shfl_sync(ACTIVE_LANES_MASK,
                reinterpret_cast<uint64_t>(unit_data_ptr), lane_empty, WARP_WIDTH);
            unit_data_ptr = reinterpret_cast<uint32_t*>(lane_empty_data_ptr);
            if (lane_id == src_lane) {

                uint32_t old_first_data =
                        atomicCAS(unit_data_ptr, EMPTY_PAIR_PTR, first_key);

                /** Branch 2.1: SUCCEED **/
                if (old_first_data == EMPTY_PAIR_PTR) {
                    // copy the remaining data
                    _Value value = atomicAdd(cnt_value_, 1);

                    CopyRemainPair(unit_data_ptr, key, value);
                    ///
                    p_mapping[value] = key_idx;
                    p_inverse_mapping[key_idx] = value;
                    ///
                    to_be_inserted = false;

                    //iterator = unit_data_ptr;
                    //mask = true;
                }

                /** Branch 2.2: failed: RESTART
                 *  In the consequent attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a different key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, *reinterpret_cast<const ptr_t*>(unit_data_ptr),
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (next_slab_ptr != EMPTY_SLAB_PTR) {
                curr_slab_ptr = next_slab_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                ptr_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_PTR_LANE) {
                    const uint32_t* unit_data_ptr =
                            get_unit_ptr_from_list_nodes(
                                              curr_slab_ptr,
                                              NEXT_SLAB_PTR_LANE);

                    ptr_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)unit_data_ptr,
                                      EMPTY_SLAB_PTR, new_next_slab_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_PTR) {
                        FreeSlab(new_next_slab_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }

//    return _make_pair(iterator, mask);
}

/////////////////

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
SearchKernel(SlabHashContext<_Key, _Value, _Hash,
                             _LOG_NUM_MEM_BLOCKS,
                             _LOG_NUM_SUPER_BLOCKS,
                             _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
             _Key* keys,
             _Value* values,
             uint8_t* founds,
             uint32_t num_queries) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= num_queries) {
        return;
    }

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_queries) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    _Pair<ptr_t*, uint8_t> result =
            slab_hash_ctx.Search(lane_active, lane_id, bucket_id, key);

    if (tid < num_queries) {
        uint8_t found = result.second;
        founds[tid] = found;
        values[tid] = found ? reinterpret_cast<_Pair<_Key, _Value>*>(result.first)
                                      ->second
                            : _Value(0);

    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
InsertAtomicKernel(SlabHashContext<_Key, _Value, _Hash,
                                   _LOG_NUM_MEM_BLOCKS,
                                   _LOG_NUM_SUPER_BLOCKS,
                                   _MEM_UNIT_WARP_MULTIPLES>
                                   slab_hash_ctx,
                   _Key* keys,
                   uint32_t num_keys,
                   uint32_t hash_coef) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.get_slab_alloc_ctx().Init(hash_coef, tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    slab_hash_ctx.InsertAtomic(lane_active, lane_id, bucket_id, key);
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
RemoveKernel(SlabHashContext<_Key, _Value, _Hash,
                             _LOG_NUM_MEM_BLOCKS,
                             _LOG_NUM_SUPER_BLOCKS,
                             _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
             _Key* keys,
             uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    slab_hash_ctx.Remove(lane_active, lane_id, bucket_id, key);
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
Search_Kernel(SlabHashContext<_Key, _Value, _Hash,
                              _LOG_NUM_MEM_BLOCKS,
                              _LOG_NUM_SUPER_BLOCKS,
                              _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
              _Key* keys,
              _Iterator<_Key, _Value>* iterators,
              uint8_t* masks,
              uint32_t num_queries) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= num_queries) {
        return;
    }

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_queries) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    _Pair<ptr_t*, uint8_t> result =
            slab_hash_ctx.Search(lane_active, lane_id, bucket_id, key);

    if (tid < num_queries) {
        iterators[tid] = reinterpret_cast<_Pair<_Key, _Value>*>(result.first);
        masks[tid] = result.second;
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
InsertAtomic_Kernel(SlabHashContext<_Key, _Value, _Hash,
                                    _LOG_NUM_MEM_BLOCKS,
                                    _LOG_NUM_SUPER_BLOCKS,
                                    _MEM_UNIT_WARP_MULTIPLES>
                                    slab_hash_ctx,
                    _Key* keys,
                    _Iterator<_Key, _Value>* iterators,
                    uint8_t* masks,
                    uint32_t num_keys,
                    uint32_t hash_coef) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.get_slab_alloc_ctx().Init(hash_coef, tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    _Pair<ptr_t*, uint8_t> result =
            slab_hash_ctx.InsertAtomic(lane_active, lane_id, bucket_id, key);

    if (tid < num_keys) {
        iterators[tid] = reinterpret_cast<_Pair<_Key, _Value>*>(result.first);
        masks[tid] = result.second;
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
Remove_Kernel(SlabHashContext<_Key, _Value, _Hash,
                              _LOG_NUM_MEM_BLOCKS,
                              _LOG_NUM_SUPER_BLOCKS,
                              _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
              _Key* keys,
              uint8_t* masks,
              uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    uint8_t success =
            slab_hash_ctx.Remove(lane_active, lane_id, bucket_id, key);

    if (tid < num_keys) {
        masks[tid] = success;
    }
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void GetIteratorsKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        ptr_t* iterators,
        uint32_t* iterator_count,
        uint32_t num_buckets) {
    // global warp ID
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t wid = tid >> 5;
    // assigning a warp per bucket
    if (wid >= num_buckets) {
        return;
    }

    /* uint32_t lane_id = threadIdx.x & 0x1F; */

    /* // initializing the memory allocator on each warp: */
    /* slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id); */

    /* uint32_t src_unit_data = */
    /*         *slab_hash_ctx.get_unit_ptr_from_list_head(wid, lane_id); */
    /* uint32_t active_mask = */
    /*         __ballot_sync(PAIR_PTR_LANES_MASK, src_unit_data !=
     * EMPTY_PAIR_PTR); */
    /* int leader = __ffs(active_mask) - 1; */
    /* uint32_t count = __popc(active_mask); */
    /* uint32_t rank = __popc(active_mask & __lanemask_lt()); */
    /* uint32_t prev_count; */
    /* if (rank == 0) { */
    /*     prev_count = atomicAdd(iterator_count, count); */
    /* } */
    /* prev_count = __shfl_sync(active_mask, prev_count, leader); */

    /* if (src_unit_data != EMPTY_PAIR_PTR) { */
    /*     iterators[prev_count + rank] = src_unit_data; */
    /* } */

    /* uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); */
    /* while (next != EMPTY_SLAB_PTR) { */
    /*     src_unit_data = */
    /*             *slab_hash_ctx.get_unit_ptr_from_list_nodes(next,
     * lane_id);
     */
    /*     count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK, */
    /*                                   src_unit_data != EMPTY_PAIR_PTR));
     */
    /*     next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); */
    /* } */
    /* // writing back the results: */
    /* if (lane_id == 0) { */
    /* } */
}

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void CountElemsKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        uint32_t* values, uint32_t* index, int* count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    uint32_t wid = tid >> 5;
    if (wid >= slab_hash_ctx.bucket_size()) {
        return;
    }

    uint32_t* src_unit_data_ptr = NULL;
    ptr_t next =
            *slab_hash_ctx.get_slab_ptr_from_list_head(wid);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
//        /*
        src_unit_data_ptr =
                slab_hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        if (NEXT_SLAB_PTR_LANE != lane_id &&
            src_unit_data_ptr[slab_hash_ctx.key_chunks + slab_hash_ctx.value_chunks - 1] != EMPTY_PAIR_PTR) {
            values[atomicAdd(index, 1)] = src_unit_data_ptr[slab_hash_ctx.key_chunks + slab_hash_ctx.value_chunks - 1];
            /*
            printf("%d %d %d %d\n", src_unit_data_ptr[3],
                                    src_unit_data_ptr[0],
                                    src_unit_data_ptr[1],
                                    src_unit_data_ptr[2]);
            */
        }

//        /*
        ///////////
        // TODO(ljm): Warning: handle protential overflow
        for (int d = 0; d != 3; ++d) {
          _Key key = reinterpret_cast<_Pair<_Key, _Value>*>(src_unit_data_ptr)->first;
          key[d] += 1;
          uint8_t lane_active = (NEXT_SLAB_PTR_LANE != lane_id) &&
                                (src_unit_data_ptr[_MEM_UNIT_WARP_MULTIPLES - 1] != EMPTY_PAIR_PTR);
          uint32_t bucket_id = slab_hash_ctx.ComputeBucket(key);
          _Pair<ptr_t*, uint8_t> result =
                slab_hash_ctx.Search(lane_active, lane_id, bucket_id, key);
          if (result.second) {
              atomicSub(count + d, 1);
              /*
              printf("found key[%d] + 1: %d\t%d\t%d --- %d\t%d\t%d\n",
                      d, key[0], key[1], key[2],
                      result.first[0], result.first[1], result.first[2]);
                      */
          }
        }
        ///////////
//        */
        next = __shfl_sync(ACTIVE_LANES_MASK, *src_unit_data_ptr, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }
}

/*
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void CountElemsPerBucketKernel(
        SlabHashContext<_Key, _Value, _Hash,
                        _LOG_NUM_MEM_BLOCKS,
                        _LOG_NUM_SUPER_BLOCKS,
                        _MEM_UNIT_WARP_MULTIPLES> slab_hash_ctx,
        uint32_t* bucket_elem_counts, uint32_t hash_coef) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    uint32_t wid = tid >> 5;
    if (wid >= slab_hash_ctx.bucket_size()) {
        return;
    }

    uint32_t count = 0;

    uint32_t src_unit_data = EMPTY_PAIR_PTR;
    ptr_t next =
            *slab_hash_ctx.get_slab_ptr_from_list_head(wid);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data =
                *slab_hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                      src_unit_data != EMPTY_PAIR_PTR));
        next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }

    // write back the results:
    if (lane_id == 0) {
        bucket_elem_counts[wid] = count;
    }
}

////////////////

template <typename _Key, typename _Value, typename _Hash,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void
BulkInsertWithMappingKernel(SlabHashContext<_Key, _Value, _Hash,
                                   _LOG_NUM_MEM_BLOCKS,
                                   _LOG_NUM_SUPER_BLOCKS,
                                   _MEM_UNIT_WARP_MULTIPLES>
                                   slab_hash_ctx,
//                   _Key* keys,
                   const int* p_coords,
                   int* p_mapping,
                   int* p_inverse_mapping,
                   uint32_t num_keys,
                   uint32_t hash_coef) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.get_slab_alloc_ctx().Init(hash_coef, tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    _Key key;

    if (tid < num_keys) {
        lane_active = true;
//        key = keys[tid];
        key = *(reinterpret_cast<const _Key*>(p_coords + tid *
                 SlabHashContext<_Key, _Value, _Hash,
                                 _LOG_NUM_MEM_BLOCKS,
                                 _LOG_NUM_SUPER_BLOCKS,
                                 _MEM_UNIT_WARP_MULTIPLES>::key_chunks));
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    slab_hash_ctx.BulkInsertWithMapping(lane_active, lane_id, bucket_id, key,
                                        p_mapping, p_inverse_mapping, tid);
    /*
    p_inverse_mapping[tid] = static_cast<int>(idx);
    p_mapping[cnt] = static_cast<int>(tid);
    */
}
////////////////
using Key = Coordinate<int, 4>;
template class SlabHash<Key, int, hash<Key>, CudaAllocator, 5, 5, 5>;
template class SlabHashContext<Key, int, hash<Key>, 5, 5, 5>;
} // namespace slab_hash
