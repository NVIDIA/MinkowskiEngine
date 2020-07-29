# Migration Guide from v0.4.x to v0.5.0

## Summary

```python
# 0.4
ME.SparseTensor(feats=feats, coords=coords, D=3)
# 0.5
ME.SparseTensor(feats=feats, coords=coords, D=3)
```


```
# 0.4
ME.MinkowskiConvolution(..., has_bias=True)
# 0.5
ME.MinkowskiConvolution(..., bias=True)
```


```
# 0.4
RegionType.HYPERCUBE
# 0.5
RegionType.HYPER_CUBE
```


## Definitions

### `CoordinateMap`

A coordinate map refers to a map object that converts a D-dimensional
coordinate into a row index for a feature matrix where the corresponding
feature for the coordinate is located. This can be implemented using
`std::map`, `std::unordered_map` or a hash-table with the right hash function
and the equality function.

### `CoordinateKey`

A `CoordinateKey` or `CoordinateMapKey` refers to a unique identifier that can
be used to retrieve a `CoordinateMap`.

### `tensor_stride`

A tensor stride is a minimum distance between non-zero elements in a sparse
tensor.  If we take a stride-2 convolution on a sparse tensor with tensor
stride 1, the resulting sparse tensor will have tensor stride 2.  If we apply
two stride-2 convolutions on a sparse tensor with tensor stride 3, the
resulting sparse tensor will have the tensor stride 2 x 2 x 3 = 12.

## From CoordsKey to CoordinateMapKey

CoordsKey should not be called in most cases, but in rare cases where you used
it. Please review this section to update your code.

One of the major difference is that we expose the pybind11 object directly to
the python side to remove the redundant abstraction layer.

In v0.4, Minkowski Engine uses a `uint64_t` hash key to identify a
`CoordinateMap`, but from v0.5, we use a tensor stride


## From CoordsManager to CoordinateManager

CoordinateManager should not be called in most cases, but if you do please re


### Initialization

```python
# 0.4.x
manager = CoordsManager(D=3)
# 0.5.x
manager = CoordinateManager(D=3)
```

## Initializing a new CoordinateMap

```python
# 0.4.x
manager = CoordsManager(D = 3)
manager.initialize(torch.IntTens
    def initialize(self,
                   coords: torch.IntTensor,
                   coords_key: CoordsKey,
                   force_creation: bool = False,
                   force_remap: bool = False,
                   allow_duplicate_coords: bool = False,
                   return_inverse: bool = False) -> torch.LongTensor:
```


## Consistent Layer Arguments
