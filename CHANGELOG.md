# Change Log

## [0.2.7] - 2019-09-04

Use `std::vector` for all internal coordinates to support arbitrary spatial dimensions.

### Changed

- Vectorized coordinates to support arbitrary spatial dimensions.
- Removed all dimension template instantiation.
- Use assertion macro for cleaner exception handling.


## [0.2.6] - 2019-08-28

Use OpenMP for multi-threaded kernel map generation and minor renaming and explicit coordinate management for future upgrades.

### Changed

- Major speed up
    - Suboptimal kernels were introduced, and optimal kernels removed for faulty cleanup in v0.2.5. CUDA kernels were re-introduced and major speed up was restored.
- Minor name changes in `CoordsManager`.
- `CoordsManager` saves all coordinates for future updates.
- `CoordsManager` functions `createInOutPerKernel` and `createInOutPerKernelTranspose` now support multi-threaded kernel map generation by default using OpenMP.
    - Thus, all manual thread functions such as `createInOutPerKernelInThreads`, `initialize_nthreads` removed.
        - Use `export OMP_NUM_THREADS` to control the number of threads.


## [0.2.5a0] - 2019-07-12

### Changed

- Added the `MinkowskiBroadcast` and `MinkowskiBroadcastConcatenation` module.


## [0.2.5] - 2019-07-02

### Changed

- Better GPU memory management:
    - GPU Memory management is now delegated to pytorch. Before the change, we need to cleanup the GPU cache that pytorch created to call `cudaMalloc`, which not only is slow but also hampers the long-running training that dies due to Out Of Memory (OOM).
