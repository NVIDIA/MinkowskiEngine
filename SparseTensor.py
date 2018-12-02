import os
import torch

import MinkowskiEngineFFI as ME
from Common import NetMetadata, convert_to_int_tensor, ffi


class SparseTensor():

    def __init__(self,
                 feats,
                 coords=None,
                 coords_key=None,
                 pixel_dist=1,
                 net_metadata=None):
        """
        Either coords or coords_key must be provided.
        pixel_distance defines the minimum space between coordinates
        """
        assert isinstance(feats, torch.Tensor), "Features must be torch.Tensor"
        assert coords is None or isinstance(
            coords, torch.IntTensor), "Coordinate must be torch.IntTensor"
        if coords is None and coords_key is None:
            raise ValueError('Coordinates or Coordinate key must be provided')
        self.C = coords
        self.F = feats
        if net_metadata is None:
            assert coords is not None, "Either provide metadata or coords"
            net_metadata = NetMetadata(coords.size(1) - 1)
        self.D = net_metadata.D
        self.use_coords_key = coords_key is not None
        self.pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        self.coords_key = coords_key if coords_key else ffi.new('uint64_t *', 0)
        self.m = net_metadata

    def stride(self, s):
        self.pixel_dist *= s

    def __add__(self, other):
        return SparseTensor(
            self.F + (other.F if isinstance(other, SparseTensor) else other),
            pixel_dist=self.pixel_dist,
            coords=self.C,
            coords_key=self.coords_key,
            net_metadata=self.m)

    def __power__(self, other):
        return SparseTensor(
            self.F**other,
            pixel_dist=self.pixel_dist,
            coords=self.C,
            coords_key=self.coords_key,
            net_metadata=self.m)

    def __repr__(self):
        if self.use_coords_key:
            return self.__class__.__name__ + '(' + os.linesep \
                + '  Feats=' + str(self.F) + os.linesep \
                + '  coords_key=' + str(self.coords_key[0]) + os.linesep \
                + '  pixel_dist=' + str(self.pixel_dist) + ')'
        else:
            return self.__class__.__name__ + '(' + os.linesep \
                + '  Feats=' + str(self.F) + os.linesep \
                + '  Coords=' + str(self.C) + os.linesep \
                + '  pixel_dist=' + str(self.pixel_dist) + ')'

    def to(self, device):
        self.F = self.F.to(device)
        return self

    def check_coords_by_coords_key(self, coords_key):
        pixel_dist = convert_to_int_tensor(0, self.D)
        exist = ME.check_coords(pixel_dist, coords_key, self.D, self.m.ffi)
        return exist

    def check_coords_by_pixel_dist(self, pixel_dist):
        """
        if the input is ffi pointer, use it as the coords_key,
        otherwise, use it as the pixel_dist.
        """
        coords_key = ffi.new('uint64_t*', 0)
        pixel_dist = convert_to_int_tensor(pixel_dist, self.D)
        exist = ME.check_coords(pixel_dist, coords_key, self.D, self.m.ffi)
        return exist

    def initialize(self):
        initialized = False
        if self.use_coords_key:
            if self.check_coords_by_coords_key(self.coords_key) > 0:
                initialized = True
        else:
            if self.check_coords_by_pixel_dist(self.pixel_dist) > 0:
                initialized = True

        if not initialized and self.C is not None:
            initialized = ME.initialize_coords(
                self.C.contiguous(), self.pixel_dist, self.D, self.m.ffi) > 0
        return initialized
