# Change Log


## [0.2.5a0] - 2019-07-12

### Changed

- Added the `MinkowskiBroadcast` and `MinkowskiBroadcastConcatenation` module.

## [0.2.5] - 2019-07-02

### Changed

- Better GPU memory management:
    - GPU Memory management is now delegated to pytorch. Before the change, we need to cleanup the GPU cache that pytorch created to call `cudaMalloc`, which not only is slow but also hampers the long-running training that dies due to Out Of Memory (OOM).
