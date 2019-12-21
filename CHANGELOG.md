# Change Log


## [nightly] - 2019-12-15

- Synchronized Batch Norm: `ME.MinkowskiSyncBatchNorm`
    - `ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm` converts a MinkowskiNetwork automatically to use synched batch norm.
- `examples/multigpu.py` update for `ME.MinkowskiSynchBatchNorm`.
- Update multigpu documentation
- Update GIL release
- Minor error fixes on `examples/modelnet40.py`
- CoordsMap size initialization updates
- Added MinkowskiUnion
- Updated MinkowskiUnion, MinkowskiPruning docs
- Use cudaMalloc instead of `at::Tensor` for GPU memory management for illegal memory access, invalid arg.
- Region hypercube iterator with even numbered kernel


## [0.3.1] - 2019-12-15

- Cache in-out mapping on device
- Robinhood unordered map for coordinate management
- hash based quantization to C++ CoordsManager based quantization with label collision
- CUDA compilation to support older devices (compute_30, 35)
- OMP_NUM_THREADS to initialize the number of threads


## [0.3.0] - 2019-12-08

- Change the hash map from google-sparsehash to Threading Building Blocks (TBB) `concurrent_unordered_map`.
    - Optional duplicate coords (CoordsMap.initialize, TODO: make mapping more user-friendly)
    - Explicit coords generation (`CoordsMap.stride`, `CoordsMap.reduce`, `CoordsMap.transposed_stride`)
    - Speed up for pooling with `kernel_size == stride_size`.
- Faster `SparseTensor.dense` function.
- Force scratch memory space to be contiguous.
- CUDA error checks
- Update Makefile
    - Architecture and sm updates for CUDA > 10.0
    - Optional cblas


## [0.2.9] - 2019-11-17

- Pytorch 1.3 support
    - Update torch cublas, cusparse handles.
- Global max pooling layers.
- Minor error fix in the coordinate manager
    - Fix cases to return `in_coords_key` when stride is identity.


## [0.2.8] - 2019-10-18

- ModelNet40 training.
- open3d v0.8 update.
- Dynamic coordinate generation.


## [0.2.7] - 2019-09-04

Use `std::vector` for all internal coordinates to support arbitrary spatial dimensions.

- Vectorized coordinates to support arbitrary spatial dimensions.
- Removed all dimension template instantiation.
- Use assertion macro for cleaner exception handling.


## [0.2.6] - 2019-08-28

Use OpenMP for multi-threaded kernel map generation and minor renaming and explicit coordinate management for future upgrades.

- Major speed up
    - Suboptimal kernels were introduced, and optimal kernels removed for faulty cleanup in v0.2.5. CUDA kernels were re-introduced and major speed up was restored.
- Minor name changes in `CoordsManager`.
- `CoordsManager` saves all coordinates for future updates.
- `CoordsManager` functions `createInOutPerKernel` and `createInOutPerKernelTranspose` now support multi-threaded kernel map generation by default using OpenMP.
    - Thus, all manual thread functions such as `createInOutPerKernelInThreads`, `initialize_nthreads` removed.
        - Use `export OMP_NUM_THREADS` to control the number of threads.


## [0.2.5a0] - 2019-07-12

- Added the `MinkowskiBroadcast` and `MinkowskiBroadcastConcatenation` module.


## [0.2.5] - 2019-07-02

- Better GPU memory management:
    - GPU Memory management is now delegated to pytorch. Before the change, we need to cleanup the GPU cache that pytorch created to call `cudaMalloc`, which not only is slow but also hampers the long-running training that dies due to Out Of Memory (OOM).
