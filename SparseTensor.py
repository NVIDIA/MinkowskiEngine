import os
import torch

from Common import convert_to_int_list, CoordsKey, CoordsManager


class SparseTensor():

    def __init__(self,
                 feats,
                 coords=None,
                 coords_key=None,
                 coords_manager=None,
                 pixel_dist=1):
        """
        Either coords or coords_key must be provided.
        pixel_distance defines the minimum space between coordinates
        coords_manager: of type CoordsManager.
        pixel_dist: distance between pixels
        """
        assert isinstance(feats, torch.Tensor), "Features must be torch.Tensor"

        if coords is None and coords_key is None:
            raise ValueError('Coordinates or Coordinate key must be provided')

        if coords_key is None:
            assert coords_manager is not None or coords is not None
            D = -1
            if coords_manager is None:
                D = coords.size(1) - 1
            else:
                D = coords_manager.D
            coords_key = CoordsKey(D)
            coords_key.setPixelDist(convert_to_int_list(pixel_dist, D))
        else:
            assert isinstance(coords_key, CoordsKey)

        if coords is not None:
            assert isinstance(coords, torch.IntTensor), \
                "Coordinate must be of type torch.IntTensor"

        if coords_manager is None:
            assert coords is not None, "Initial coordinates must be given"
            D = coords.size(1) - 1
            coords_manager = CoordsManager(D)
            coords_manager.initialize(coords, coords_key)
        else:
            assert isinstance(coords_manager, CoordsManager)

        self.F = feats
        self.coords_key = coords_key
        self.coords_man = coords_manager

    @property
    def pixel_dist(self):
        return self.coords_key.getPixelDist()

    @property
    def C(self):
        return self.coords_man

    @property
    def D(self):
        return self.coords_key.D

    def stride(self, s):
        ss = convert_to_int_list(s)
        pixel_dists = self.coords_key.getPixelDist()
        self.coords_key.setPixelDist([s * p for s, p in zip(ss, pixel_dists)])

    def __add__(self, other):
        return SparseTensor(
            self.F + (other.F if isinstance(other, SparseTensor) else other),
            coords_key=self.coords_key,
            coords_manager=self.C)

    def __power__(self, other):
        return SparseTensor(
            self.F**other, coords_key=self.coords_key, coords_manager=self.C)

    def __repr__(self):
        return self.__class__.__name__ + '(' + os.linesep \
            + '  Feats=' + str(self.F) + os.linesep \
            + '  coords_key=' + str(self.coords_key) + os.linesep \
            + '  pixel_dist=' + str(self.coords_key.getPixelDist()) + os.linesep \
            + '  coords_man=' + str(self.coords_man) + ')'

    def to(self, device):
        self.F = self.F.to(device)
        return self

    def getKey(self):
        return self.coords_key
