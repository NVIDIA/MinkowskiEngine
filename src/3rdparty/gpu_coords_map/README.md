# Multi-thread GPU CoordinateHash with Shared SLAB Router
Multi-thread version of GPU CoordinateHash with shared slab router

## What's news?
- Light Head: the table head has been shrinked 32 times.
- Singleton: reuse slab router by making SlabAlloc singleton.
- Multi-thread: one slab router, multi table head.
- Random Hash: one random number per table head which reduces the
   confliction.

## Usage example:
more details in test_unique_with_remove_multithread.cu

```
int main() {
  std::vector<std::thread> vt;
  vt.reserve(50);
  for (int i = 0; i != 50; ++i) {
      vt.emplace_back(std::thread([i] {
          TEST_6DIM_KEYS_THRUST(1000000);
          std::cout << "Finish " << i << "th TEST_6DIM_KEYS_THRUST" << std::endl;
      }));
  }

  for (int i = 0; i != 50; ++i) {
      vt[i].join();
  }
}
```

##TODO


1. General improvment:

  - reduce memory Alloc and memory copy times.
  - cuda memory pool for key value storage.

2. Custom it to specific usage:

  - custom kernel
  - custom memory handling

---------------------------------

# GPU CoordinateHash
This is a modified version of [SlabHash](https://github.com/owensgroup/SlabHash).

Original SlabHash only supports <uint32_t, uint32_t> key value pairs.
Now in theory it supports arbitrary value types, and multi dimensional <int, long, float, double> keys. It also supports self-defined hash function in template.

At current only `Key<uint32_t, 7>, Value<uint32_t>` was tested.

## Publication
This library is based on the original slab hash paper, initially proposed in the following IPDPS'18 paper:
* [Saman Ashkiani, Martin Farach-Colton, John Owens, *A Dynamic Hash Table for the GPU*, 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)](https://ieeexplore.ieee.org/abstract/document/8425196)

This library is a rafactored and slightly redesigned version of the original code, so that it can be extended and be used in other research projects as well. It is still under continuous development. If you find any problem with the code, or suggestions for potential additions to the library, we will appreciate it if you can raise issues on github. We will address them as soon as possible. 

## Compilation
1. Make sure to edit `CMakeLists.txt` such that it reflects the GPU device's compute capability. For example, to include compute 3.5 you should have `option(SLABHASH_GENCODE_SM35 "GENCODE_SM35" ON)`.
2. `mkdir build && cd build`
3. `cmake ..`
4. `make -j4`

## Usage
It is now a header only library. Include `coordinate_hash_map.cuh` or `coordinate_indexer.cuh` in your .cu file to use the lib. Documents TBD.

## TODO
- Update copyrights to be consistent with the original Apache license from [SlabHash](https://github.com/owensgroup/SlabHash).
- Include parallel iterators.
- Add pybind and improve `touch/allocate` function for voxel hashing.
