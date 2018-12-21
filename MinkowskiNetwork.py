from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import convert_to_int_tensor


class MinkowskiNetwork(nn.Module, ABC):
    """
    MinkowskiNetwork: an abstract class for sparse convnets.

    Note: All modules that use the same coordinates must use the same net_metadata
    """

    def __init__(self, D):
        super(MinkowskiNetwork, self).__init__()
        self.D = D

    @abstractmethod
    def forward(self, x):
        pass

    def init(self, x):
        """
        Initialize coordinates if it does not exist
        """
        nrows = self.get_nrows(1)
        if nrows < 0:
            if isinstance(x, SparseTensor):
                self.initialize_coords(x.C)
            else:
                raise ValueError('Initialize input coordinates')
        elif nrows != x.F.size(0):
            raise ValueError('Input size does not match the coordinate size')

    def get_permutation(self, pixel_dist_src, pixel_dist_dst):
        pixel_dist_src = convert_to_int_tensor(pixel_dist_src, self.D)
        pixel_dist_dst = convert_to_int_tensor(pixel_dist_dst, self.D)
        assert pixel_dist_src.numel() == pixel_dist_dst.numel()

        # Mapping is surjective, Mapping to smaller space is not supported
        for i in range(pixel_dist_src.numel()):
            assert pixel_dist_src[i] >= pixel_dist_dst[i]

        perm = torch.IntTensor()
        success = MEB.get_permutation(perm, pixel_dist_src, pixel_dist_dst,
                                      self.D, self.net_metadata.ffi)
        if success < 0:
            raise ValueError('get_permutation failed')
        return perm

    def get_index_map(self, coords, pixel_dist):
        """
        Get the current coords (with duplicates) index map.

        If pixel_dist > 1, use
        coords = torch.cat(((coords[:, :D] / pixel_dist) * pixel_dist,
                            coords[:, D:]), dim=1)
        """
        assert isinstance(coords, torch.IntTensor), "Coord must be IntTensor"
        index_map = torch.IntTensor()
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        success = MEB.get_index_map(coords.contiguous(), index_map, pixel_dist,
                                    self.D, self.net_metadata.ffi)
        if success < 0:
            raise ValueError('get_index_map failed')
        return index_map

    def permute_label(self, label, max_label, pixel_dist):
        if pixel_dist == 1 or np.prod(pixel_dist) == 1:
            return label

        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        permutation = self.get_permutation(pixel_dist, 1)
        nrows = self.get_nrows(pixel_dist)

        label = label.contiguous().numpy()
        permutation = permutation.numpy()

        counter = np.zeros((nrows, max_label), dtype='int32')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def permute_feature(self, feat, pixel_dist, dtype=np.float32):
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        permutation = self.get_permutation(pixel_dist, 1)
        nrows = self.get_nrows(pixel_dist)

        feat_np = feat.contiguous().numpy()
        warped_feat = np.zeros((nrows, feat.size(1)), dtype=dtype)
        counter = np.zeros((nrows, 1), dtype='int32')
        for j in range(feat.size(1)):
            np.add.at(warped_feat, (permutation, j), feat_np[:, j])
        np.add.at(counter, permutation, 1)
        warped_feat = warped_feat / counter
        return torch.from_numpy(warped_feat)
