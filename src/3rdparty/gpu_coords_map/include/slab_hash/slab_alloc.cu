#include <vector>
#include <thrust/device_vector.h>
#include "slab_alloc.h"

namespace slab_hash {

template <class _Alloc,
          uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
std::vector<int> SlabAlloc<_Alloc, _LOG_NUM_MEM_BLOCKS,
                                   _LOG_NUM_SUPER_BLOCKS,
                                   _MEM_UNIT_WARP_MULTIPLES>::
    CountSlabsPerSuperblock() {
        const uint32_t num_super_blocks = slab_alloc_context_.num_super_blocks_;

        auto slabs_per_superblock_buffer =
                allocator_->template allocate<uint32_t>(num_super_blocks);
        thrust::device_vector<uint32_t> slabs_per_superblock(
                slabs_per_superblock_buffer,
                slabs_per_superblock_buffer + num_super_blocks);
        thrust::fill(slabs_per_superblock.begin(), slabs_per_superblock.end(),
                     0);

        // counting total number of allocated memory units:
        int blocksize = 128;
        int num_mem_units =
                slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
        int num_cuda_blocks = (num_mem_units + blocksize - 1) / blocksize;
        CountSlabsPerSuperblockKernel<_LOG_NUM_MEM_BLOCKS,
                                      _LOG_NUM_SUPER_BLOCKS,
                                      _MEM_UNIT_WARP_MULTIPLES><<<num_cuda_blocks, blocksize>>>(
                slab_alloc_context_,
                thrust::raw_pointer_cast(slabs_per_superblock.data()));

        std::vector<int> result(num_super_blocks);
        thrust::copy(slabs_per_superblock.begin(), slabs_per_superblock.end(),
                     result.begin());
        allocator_->template deallocate<uint32_t>(slabs_per_superblock_buffer);
        return std::move(result);
    }
template <uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
    __device__ uint32_t SlabAllocContext<_LOG_NUM_MEM_BLOCKS,
                                         _LOG_NUM_SUPER_BLOCKS,
                                         _MEM_UNIT_WARP_MULTIPLES>::
    WarpAllocate(const uint32_t& lane_id) {
        // tries and allocate a new memory units within the resident memory
        // block if it returns 0xFFFFFFFF, then there was not any empty memory
        // unit a new resident block should be chosen, and repeat again
        // allocated result:  _LOG_NUM_SUPER_BLOCKS  bits: super_block_index
        //                    (22 - _LOG_NUM_SUPER_BLOCKS) bits: memory block index
        //                    5  bits: memory unit index (hi-bits of 10bit)
        //                    5  bits: memory unit index (lo-bits of 10bit)
        int empty_lane = -1;
        uint32_t free_lane;
        uint32_t read_bitmap = resident_bitmap_;
        uint32_t allocated_result = 0xFFFFFFFF;
        // works as long as <31 bit are used in the allocated_result
        // in other words, if there are 32 super blocks and at most 64k blocks
        // per super block

        while (allocated_result == 0xFFFFFFFF) {
            empty_lane = __ffs(~resident_bitmap_) - 1;
            free_lane = __ballot_sync(0xFFFFFFFF, empty_lane >= 0);
            if (free_lane == 0) {
                // all bitmaps are full: need to be rehashed again:
                updateMemBlockIndex(((threadIdx.x + blockIdx.x * blockDim.x) >>
                                    5) + hash_coef_);
                read_bitmap = resident_bitmap_;
                continue;
            }
            uint32_t src_lane = __ffs(free_lane) - 1;
            if (src_lane == lane_id) {
                read_bitmap = atomicCAS(
                        super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                                resident_index_ * BITMAP_SIZE_ + lane_id,
                        resident_bitmap_, resident_bitmap_ | (1 << empty_lane));
                if (read_bitmap == resident_bitmap_) {
                    // successful attempt:
                    resident_bitmap_ |= (1 << empty_lane);
                    allocated_result =
                            (super_block_index_
                             << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                            (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                            (lane_id << MEM_UNIT_BIT_OFFSET_ALLOC_) |
                            empty_lane;
                } else {
                    // Not successful: updating the current bitmap
                    resident_bitmap_ = read_bitmap;
                }
            }
            // asking for the allocated result;
            allocated_result =
                    __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
        }
        return allocated_result;
    }

    // called when the allocator fails to find an empty unit to allocate:
template <uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
    __device__ void SlabAllocContext<_LOG_NUM_MEM_BLOCKS,
                                     _LOG_NUM_SUPER_BLOCKS,
                                     _MEM_UNIT_WARP_MULTIPLES>::updateMemBlockIndex(uint32_t global_warp_id) {
        num_attempts_++;
        assert(num_attempts_ < 11);
        super_block_index_++;
        super_block_index_ = (super_block_index_ == num_super_blocks_)
                                     ? 0
                                     : super_block_index_;

        resident_index_++;
        resident_index_ = (resident_index_ == NUM_MEM_BLOCKS_PER_SUPER_BLOCK_)
                                     ? 0
                                     : resident_index_;

        // loading the assigned memory block:
        resident_bitmap_ =
                *((super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_) +
                  resident_index_ * BITMAP_SIZE_ + (threadIdx.x & 0x1F));
    }

template <uint32_t _LOG_NUM_MEM_BLOCKS,
          uint32_t _LOG_NUM_SUPER_BLOCKS,
          uint32_t _MEM_UNIT_WARP_MULTIPLES>
__global__ void CountSlabsPerSuperblockKernel(SlabAllocContext<_LOG_NUM_MEM_BLOCKS,
                                                               _LOG_NUM_SUPER_BLOCKS,
                                                               _MEM_UNIT_WARP_MULTIPLES> context,
                                              uint32_t* slabs_per_superblock) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int num_bitmaps = context.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
    if (tid >= num_bitmaps) {
        return;
    }

    for (int i = 0; i < context.num_super_blocks_; i++) {
        uint32_t read_bitmap = *(context.get_ptr_for_bitmap(i, tid));
        atomicAdd(&slabs_per_superblock[i], __popc(read_bitmap));
    }
}

template class SlabAlloc<CudaAllocator, 5, 5, 5>;
template class SlabAllocContext<5, 5, 5>;

//template __device__ uint32_t SlabAllocContext<5, 5, 5>::WarpAllocate(const uint32_t& lane_id);

} // namespace slab_hash
