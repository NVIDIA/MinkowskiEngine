#include "cuda_unordered_map.h"
#include "coordinate.h"

namespace cuda {
//////////////
template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::BulkInsert(const int* p_coords,
                   int* p_mapping,
                   int* p_inverse_mapping,
                   int size,  int key_chunks_) {
  assert(key_chunks_ == key_chunks);
  slab_hash_->BulkInsertWithMapping(p_coords, p_mapping,
                                    p_inverse_mapping, size);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateKeys(int* p_coords, int size) {
  slab_hash_->IterateKeys(p_coords, size);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateSearchAtBatch(int* p_out, int batch_index, int size) {
  slab_hash_->IterateSearchAtBatch(p_out, batch_index, size);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateSearchPerBatch(const std::vector<int*>& p_outs, int size) {
  slab_hash_->IterateSearchPerBatch(p_outs, size);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateOffsetInsert(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                         int* p_offset, int size) {
  slab_hash_->IterateOffsetInsert(in_map->get_slab_hash(),
                                  p_offset, size);
}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateOffsetInsertWithInsOuts(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    int* p_offset,
                                    int* p_in, int* p_out,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateOffsetSearch(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    int* p_offset,
                                    int* p_in, int* p_out,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateBatchInsert(const std::shared_ptr<unordered_map<Key, Value,
                                            LOG_NUM_SUPER_BLOCKS,
                                            LOG_NUM_MEM_BLOCKS,
                                            Hash, Alloc>>& in_map,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateBatchSearch(const std::shared_ptr<unordered_map<Key, Value,
                                            LOG_NUM_SUPER_BLOCKS,
                                            LOG_NUM_MEM_BLOCKS,
                                            Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateStrideInsert(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    const std::vector<int>& tensor_strides,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateStrideInsertWithInOut(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                    const std::vector<int>& tensor_strides,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateStrideSearch(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                    const std::vector<int>& tensor_strides,
                                    int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateInsert(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                             int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateInsertWithInsOuts(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                             int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IterateSearch(const std::shared_ptr<unordered_map<Key, Value,
                                             LOG_NUM_SUPER_BLOCKS,
                                             LOG_NUM_MEM_BLOCKS,
                                             Hash, Alloc>>& in_map,
                                    int* p_in, int* p_out,
                                             int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IteratePruneInsert(const std::shared_ptr<unordered_map<Key, Value,
                                                  LOG_NUM_SUPER_BLOCKS,
                                                  LOG_NUM_MEM_BLOCKS,
                                                  Hash, Alloc>>& in_map,
                                bool* p_keep, int keep_size,
                                int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IteratePruneInsertWithInOut(const std::shared_ptr<unordered_map<Key, Value,
                                                  LOG_NUM_SUPER_BLOCKS,
                                                  LOG_NUM_MEM_BLOCKS,
                                                  Hash, Alloc>>& in_map,
                                int* p_in, int* p_out,
                                bool* p_keep, int keep_size,
                                int size) {}

template <typename Key, typename Value,
          uint32_t LOG_NUM_SUPER_BLOCKS,
          uint32_t LOG_NUM_MEM_BLOCKS,
          typename Hash, class Alloc>
void
unordered_map<Key, Value,
              LOG_NUM_SUPER_BLOCKS,
              LOG_NUM_MEM_BLOCKS,
              Hash, Alloc>::
IteratePruneSearch(const std::shared_ptr<unordered_map<Key, Value,
                                                  LOG_NUM_SUPER_BLOCKS,
                                                  LOG_NUM_MEM_BLOCKS,
                                                  Hash, Alloc>>& in_map,
                                int* p_in, int* p_out,
                                bool* p_keep, int keep_size,
                                int size) {}
//////////////

template class unordered_map<Coordinate<int, 4>, int, 5, 5>;

} // cuda
