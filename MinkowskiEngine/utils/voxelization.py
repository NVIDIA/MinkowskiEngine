import numpy as np

import MinkowskiEngineBackend as MEB


def fnv_hash_vec(arr):
    """
    FNV64-1A

    Given a numpy array of N X D, generate hash values using the same hash
    function used for ME
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = np.floor(arr).astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
        np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    FNV64-1A

    Given a numpy array of N X D, generate hash values using the same hash
    function used for ME
    """
    assert arr.ndim == 2
    arr = np.floor(arr)
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64)

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def SparseVoxelize(coords,
                   feats=None,
                   labels=None,
                   ignore_label=255,
                   return_index=False,
                   hash_type='ravel'):
    """
    Given coordinates, and features (optionally labels), generate voxelized
    coords, features (and labels when given).
    """
    use_label = labels is not None
    use_feat = feats is not None
    assert hash_type in [
        'ravel', 'fnv'
    ], f"Invalid hash_type. Either ravel, or fnv allowed. You put hash_type={hash_type}"
    assert coords.ndim == 2
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize
    if hash_type == 'ravel':
        key = ravel_hash_vec(coords)
    else:
        key = fnv_hash_vec(coords)

    if use_label:
        inds, labels = MEB.SparseVoxelization(key, labels.astype(np.int32),
                                              ignore_label, use_label)
        if return_index:
            return inds, labels
        else:
            return coords[inds], feats[inds], labels
    else:
        _, inds = np.unique(key, return_index=True)
        if return_index:
            return inds
        else:
            if use_feat:
                return coords[inds], feats[inds]
            else:
                return coords[inds]
