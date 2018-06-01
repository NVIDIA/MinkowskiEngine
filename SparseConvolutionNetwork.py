from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn

import SparseConvolutionEngineFFI as SCE
from Common import Metadata, convert_to_int_tensor


class SparseConvolutionNetwork(nn.Module, ABC):
    """
    SparseConvolutionNetwork: an abstract class for sparse convnets.

    Note: All modules that use the same coordinates must use the same metadata
    """

    def __init__(self, D):
        super(SparseConvolutionNetwork, self).__init__()
        self.D = D
        self.metadata = Metadata(D)

    @abstractmethod
    def forward(self, x):
        if self.n_rows < 0:
            raise ValueError('Initialize input coordinates')
        elif self.n_rows != x.size(0):
            raise ValueError('Input size does not match the coordinate size')

    def clear(self):
        self.metadata.clear()

    def initialize_coords(self, coords):
        pixel_dist = convert_to_int_tensor(1, self.D)
        SCE.initialize_coords(coords.contiguous(), pixel_dist, self.D,
                              self.metadata.ffi)
        self.n_rows = coords.size(0)

    def initialize_coords_with_duplicates(self, coords):
        pixel_dist = convert_to_int_tensor(1, self.D)
        SCE.initialize_coords_with_duplicates(
            coords.contiguous(), pixel_dist, self.D, self.metadata.ffi)
        self.n_rows = self.get_nrows(1)

    def get_coords(self, pixel_dist):
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        coords = torch.IntTensor()
        success = SCE.get_coords(coords, pixel_dist, self.D, self.metadata.ffi)
        if success < 0:
            raise ValueError('No coord found at : {}'.format(pixel_dist))
        return coords

    def get_permutation(self, pixel_dist_src, pixel_dist_dst):
        pixel_dist_src = convert_to_int_tensor(pixel_dist_src, self.D)
        pixel_dist_dst = convert_to_int_tensor(pixel_dist_dst, self.D)
        assert pixel_dist_src.numel() == pixel_dist_dst.numel()

        # Mapping is surjective, Mapping to smaller space is not supported
        for i in range(pixel_dist_src.numel()):
            assert pixel_dist_src[i] > pixel_dist_dst[i]

        perm = torch.IntTensor()
        success = SCE.get_permutation(perm, pixel_dist_src, pixel_dist_dst,
                                      self.D, self.metadata.ffi)
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
        index_map = torch.IntTensor()
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        success = SCE.get_index_map(coords.contiguous(), index_map, pixel_dist,
                                    self.D, self.metadata.ffi)
        if success < 0:
            raise ValueError('get_index_map failed')
        return index_map

    def get_nrows(self, pixel_dist):
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        nrows = SCE.get_nrows(pixel_dist, self.D, self.metadata.ffi)
        if nrows < 0:
            raise ValueError('No coord found at : {}'.format(pixel_dist))
        return nrows

    def permute_label(self, label, max_label, pixel_dist):
        if pixel_dist == 1 or np.prod(pixel_dist) == 1:
            return label

        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        permutation = self.get_permutation(pixel_dist, 1)
        nrows = self.get_nrows(pixel_dist)

        label = label.contiguous().numpy()
        permutation = permutation.numpy()

        counter = np.zeros((nrows, max_label), dtype='int')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def permute_feature(self, feat, pixel_dist, dtype=np.float32):
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        permutation = self.get_permutation(pixel_dist, 1)
        nrows = self.get_nrows(pixel_dist)

        feat_np = feat.contiguous().numpy()
        warped_feat = np.zeros((nrows, feat.size(1)), dtype=dtype)
        counter = np.zeros((nrows, 1), dtype='int')
        for j in range(feat.size(1)):
            np.add.at(warped_feat, (permutation, j), feat_np[:, j])
        np.add.at(counter, permutation, 1)
        warped_feat = warped_feat / counter
        return torch.from_numpy(warped_feat)
