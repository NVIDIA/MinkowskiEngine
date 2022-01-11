# Change Log

## [0.5.5]

- MKL compilation fix (#358)
- Example updates for `has_bias` (PR #356)
- Add thrust exception handling (issue #357)
- `min_coordinate` to device in `dense()`
- `cpu_kernel_region` `to_gpu` now uses `shared_ptr` allocator for gpu memory
- Docker installation instruction added
- Fix for GPU `coo_spmm` when nnz == 0
    - Fix MinkowskiInterpolationGPU for invalid samples (issue #383)
- gradcheck wrap func 1.9
- Handles `dense` for an empty tensor (issue #384)
- Handles an emtpy tensor for convolution (issue #384)
- When `coordinates` is provided in `MinkowskiToSparseTensor`, `remove_zeros` will be ignored (issue #387)
- Pybind equality error from the package (issue #414)
- Fix undefined coordinate merge for multiple coordinate unions
- Add cross-shaped kernel support (issue #436)

## [0.5.4]

- Fix `TensorField.sparse()` for no duplicate coordinates
- Skip unnecessary spmm if `SparseTensor.initialize_coordinates()` has no duplicate coordinates
- Model summary utility function added
- TensorField.splat function for splat features to a sparse tensor
- SparseTensor.interpolate function for extracting interpolated features
- `coordinate_key` property function for `SparseTensor` and `TensorField`
- Fix .dense() for GPU tensors. (PR #319)

## [0.5.3]

- Updated README for pytorch 1.8.1 support
- Use custom `gpu_storage` instead of thrust vector for faster constructors
- pytorch installation instruction updates
- fix transpose kernel map with `kernel_size == stride_size`
- Update reconstruction and vae examples for v0.5 API
- `stack_unet.py` example, API updates
- `MinkowskiToFeature` layer

## [0.5.2]

- spmm average cuda function
- SparseTensor list operators (cat, mean, sum, var)
- MinkowskiStack containers
- Replace all at::cuda::getCurrentCUDASparseHandle with custom getCurrentCUDASparseHandle (issue #308)
- fix coordinate manager kernel map python function
- direct max pool
    - SparseTensorQuantizationMode.MAX_POOL
- TensorField global max pool
    - origin field
    - origin field map
    - MinkowskiGlobalMaxPool CPU/GPU updates for a field input
- SparseTensor.dense() raises a value error when a coordinate is negative rather than subtracting the minimum coordinate from a sparse tensor. (issue #316)
- Added `to_sparse()` that removes zeros. (issue #317)
    - Previous `to_sparse()` was renamed to `to_sparse_all()`
    - `MinkowskiToSparseTensor` takes an optional `remove_zeros` boolean argument.
- Fix global max pool with batch size 1
- Use separate memory chunks for in, out map, and kernel indices for `gpu_kernel_map` for gpu memory misaligned error


## [0.5.1]

- v0.5 documentation updates
- Nonlinear functionals and modules
- Warning when using cuda without ME cuda support
- diagnostics test
- TensorField slice
    - Cache the unique map and inverse map pair in the coordinate manager
    - generate inverse_mapping on the fly
- CoordinateManager
    - `field_to_sparse_insert_and_map`
    - `exists_field_to_sparse`
    - `get_field_to_sparse_map`
    - fix `kernel_map` with empty coordinate maps
- CoordiateFieldMap
    - `quantize_coordinates`
- TensorField binary ops fix
- MinkowskiSyncBatchNorm
    - Support tfield
    - conver sync batchnorm updates
- TensorField to sparse with coordinate map key
- Sparse matrix multiplication
    - force contiguous matrix
- Fix AveragePooling cudaErrorMisalignedAddress error for CUDA 10 (#246)

## [0.5.0] - 2020-12-24

## [0.5.0a] - 2020-08-05

### Changed

- Remove Makefile for installation as pytorch supports multithreaded compilation
- GPU coordinate map support
- Coordinate union
- Sparse tensor binary operators
- CUDA 11.1 support
- quantization function updates
- Multi GPU examples
- Pytorch-lightning multi-gpu example
- Transpose pooling layers
- TensorField updates
- Batch-wise decomposition
- inverse_map when sparse() called (slice)
- ChannelwiseConvolution
- TensorField support for non-linearities

## [0.4.3] - 2020-05-29

### Changed

- Use `CPU_ONLY` compile when `torch` fails to detect a GPU (Issue #105)
- Fix `get_kernel_map` for `CPU_ONLY` (Issue #107)
- Update `get_union_map` doc (Issue #108)
- Abstract getattr minkowski backend functions
- Add `coordinates_and_features_at(batch_index)` function in the SparseTensor class.
- Add `MinkowskiChannelwiseConvolution` (Issue #92)
- Update `MinkowskiPruning` to generate an empty sparse tensor as output (Issue #102)
- Add `return_index` for `sparse_quantize`
- Templated CoordsManager for coords to int and coords to vector classes
- Sparse tensor quantization mode
    - Features at duplicated coordinates will be averaged automatically with `quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE`
- `SparseTensor.slice()` slicing features on discrete coordinates to continuous coordinates
- CoordsManager.getKernelMapGPU returns long type tensors (Issue #125)
- SyncBatchNorm error fix (Issue #129)
- Sparse Tensor `dense()` doc update (Issue #126)
- Installation arguments `--cuda_home=<value>`, `--force_cuda`, `--blas_include_dirs=<comma_separated_values>`, and '--blas_library_dirs=<comma_separated_values>`. (Issue #135)
- SparseTensor query by coordinates `features_at_coords` (Issue #137)
- Memory manager control. CUDA | Pytorch memory manager for cuda malloc


## [0.4.2] - 2020-03-13

### Added

- Completion and VAE examples
- GPU version of getKernelMap: getKernelMapGPU

### Changed

- Fix dtype double to float on the multi-gpu example
- Remove the dimension input argument on GlobalPooling, Broadcast functions
- Kernel map generation has tensor stride > 0 check
- Fix `SparseTensor.set_tensor_stride`
- Track whether the batch indices are set first when initializing coords, The initial batch indices will be used throughout the lifetime of a sparse tensor
- Add a memory warning on ModelNet40 training example (Issue #86)
- Update the readme, definition
- Fix an error in examples.convolution
- Changed `features_at`, `coordinates_at` to take a batch index not the index of the unique batch indices. (Issue #100)
- Fix an error torch.range --> torch.arange in `sparse_quantize` (Issue #101)
- Fix BLAS installation link error (Issue #94)
- Fix `MinkowskiBroadcast` and `MinkowskiBroadcastConcatenation` to use arbitrary channel sizes
- Fix `pointnet.py` example (Issue #103)


## [0.4.1] - 2020-01-28

### Changed

- Kernel maps with region size 1 do not require `Region` class initialization.
- Faster union map with out map initialization
- Batch index order hot fix on `dense()`, `sparse()`


## [0.4.0] - 2020-01-26

### Added

- Add `MinkowskiGlobalSumPooling`, `MinkowskiGlobalAvgPooling`
- Add `examples/convolution.py` to showcase various usages
- Add `examples/sparse_tensor_basic.py` and a SparseTensor tutorial page
- Add convolution, kernel map gifs
- Add batch decomposition functions
    - Add `SparseTensor.decomposed_coordinates`
    - Add `SparseTensor.decomposed_features`
    - Add `SparseTensor.coordinates_at(batch_index)`
    - Add `SparseTensor.features_at(batch_index)`
    - Add `CoordsManager.get_row_indices_at(coords_key, batch_index)`


### Changed

- `SparseTensor` additional coords.device guard
- `MinkowskiConvolution`, `Minkowski*Pooling` output coordinates will be equal to the input coordinates if stride == 1. Before this change, they generated output coordinates previously defined for a specific tensor stride.
- `MinkowskiUnion` and `Ops.cat` will take a variable number of sparse tensors not a list of sparse tensors
- Namespace cleanup
- Fix global in out map with uninitialized global map
- `getKernelMap` now can generate new kernel map if it doesn't exist
- `MinkowskiPruning` initialization takes no argument
- Batched coordinates with batch indices prepended before coordinates


## [0.3.3] - 2020-01-07

### Added

- Add `get_coords_map` on `CoordsManager`.
- Add `get_coords_map` on `MinkowskiEngine.utils`.
- Sparse Tensor Sparse Tensor binary operations `(+,-,*,/)`
    - Binary operations between sparse tensors or sparse tensor + pytorch tensor
    - Inplace operations for the same coords key
- Sparse Tensor operation mode
    - Add `set_sparse_tensor_operation_mode` sharing the global coords manager by default

### Changed

- Minor changes on `setup.py` for torch installation check and system assertions.
- Update BLAS installation configuration.
- Update union kernel map and union coords to use reference wrappers.
- namespace `minkowski` for all cpp, cu files
- `MinkowskiConvolution` and `MinkowskiConvolutionTranspose` now support output coordinate specification on the function call.
- `Minkowski[Avg|Max|Sum]Pooling` and `Minkowski[Avg|Max|Sum]PoolingTranspose` now support output coordinate specification on the function call.


## [0.3.2] - 2019-12-25

### Added
- Synchronized Batch Norm: `ME.MinkowskiSyncBatchNorm`
    - `ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm` converts a MinkowskiNetwork automatically to use synched batch norm.
- `examples/multigpu.py` update for `ME.MinkowskiSynchBatchNorm`.
- Add `MinkowskiUnion`
- Add CoordsManager functions
    - `get_batch_size`
    - `get_batch_indices`
    - `set_origin_coords_key`
- Add `quantize_th`, `quantize_label_th`
- Add MANIFEST

### Changed

- Update `MinkowskiUnion`, `MinkowskiPruning` docs
- Update multigpu documentation
- Update GIL release
- Use cudaMalloc instead of `at::Tensor` for GPU memory management for illegal memory access, invalid arg.
- Minor error fixes on `examples/modelnet40.py`
- CoordsMap size initialization updates
- Region hypercube iterator with even numbered kernel
- Fix global reduction in-out map with non contiguous batch indices
- GlobalPooling with torch reduction
- Update CoordsManager function `get_row_indices_per_batch` to return a list of `torch.LongTensor` for mapping indices. The corresponding batch indices is accessible by `get_batch_indices`.
- Update `MinkowskiBroadcast`, `MinkowskiBroadcastConcatenation` to use row indices per batch (`getRowIndicesPerBatch`)
- Update `SparseTensor`
    - `allow_duplicate_coords` argument support
    - update documentation, add unittest
- Update the training demo and documentation.
- Update `MinkowskiInstanceNorm`: no `dimension` argument.
- Fix CPU only build


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
