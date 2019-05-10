import torch

import unittest
import numpy as np

import MinkowskiEngineBackend as MEB
from MinkowskiEngine.utils import ravel_hash_vec, sparse_quantize


class TestGPUVoxelization(unittest.TestCase):

    def test(self):
        N = 16575
        ignore_label = 255

        coords = np.random.rand(N, 3) * 100
        feats = np.random.rand(N, 4)
        labels = np.floor(np.random.rand(N) * 3)

        labels = labels.astype(np.int32)

        # Make duplicates
        coords[:3] = 0
        labels[:3] = 2

        key = ravel_hash_vec(coords)  # floor happens by astype(np.uint64)

        inds, labels_v = MEB.SparseVoxelization(
            key, labels.astype(np.int32), ignore_label, True)
        coords_v, feats_v = coords[inds], feats[inds]
        print(coords_v, feats_v)

        outputs = sparse_quantize(coords, feats, labels, ignore_label)
        print(outputs)


if __name__ == '__main__':
    unittest.main()
