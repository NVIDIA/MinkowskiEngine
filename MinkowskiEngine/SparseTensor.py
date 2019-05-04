import os
import torch

from Common import convert_to_int_list
from MinkowskiCoords import CoordsKey, CoordsManager


class SparseTensor():
    r"""A sparse tensor class.

    """

    def __init__(self,
                 feats,
                 coords=None,
                 coords_key=None,
                 coords_manager=None,
                 tensor_stride=1):
        r"""
        Either coords or coords_key must be provided.
        coords_manager: of type CoordsManager.
        tensor_stride: defines the minimum distance or stride between coordinates
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
            coords_key.setTensorStride(convert_to_int_list(tensor_stride, D))
        else:
            assert isinstance(coords_key, CoordsKey)

        if coords is not None:
            assert isinstance(coords, torch.IntTensor), \
                "Coordinate must be of type torch.IntTensor"

        if coords_manager is None:
            assert coords is not None, "Initial coordinates must be given"
            D = coords.size(1) - 1
            coords_manager = CoordsManager(D=D)
            coords_manager.initialize(coords, coords_key)
        else:
            assert isinstance(coords_manager, CoordsManager)

        self._F = feats.contiguous()
        self._C = coords
        self.coords_key = coords_key
        self.coords_man = coords_manager

    @property
    def tensor_stride(self):
        return self.coords_key.getTensorStride()

    @tensor_stride.setter
    def tensor_stride(self, p):
      """
      The function is not recommended to be used directly.
      """
      p = convert_to_int_list(p, self.D)
      self.coords_key.setTensorStride(p)

    def _get_coords(self):
        return self.coords_man.get_coords(self.coords_key)

    @property
    def C(self):
        return self.coords

    @property
    def coords(self):
        if self._C is None:
            self._C = self._get_coords()
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def feats(self):
        return self._F

    @property
    def D(self):
        return self.coords_key.D

    def stride(self, s):
        ss = convert_to_int_list(s)
        tensor_strides = self.coords_key.getTensorStride()
        self.coords_key.setTensorStride(
            [s * p for s, p in zip(ss, tensor_strides)])

    def __add__(self, other):
        return SparseTensor(
            self._F + (other.F if isinstance(other, SparseTensor) else other),
            coords_key=self.coords_key,
            coords_manager=self.coords_man)

    def __power__(self, power):
        return SparseTensor(
            self._F**power, coords_key=self.coords_key, coords_manager=self.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '(' + os.linesep \
            + '  Feats=' + str(self.F) + os.linesep \
            + '  coords_key=' + str(self.coords_key) + os.linesep \
            + '  tensor_stride=' + str(self.coords_key.getTensorStride()) + os.linesep \
            + '  coords_man=' + str(self.coords_man) + ')'

    def __len__(self):
        return len(self._F)

    def size(self):
        return self._F.size()

    def to(self, device):
        self._F = self._F.to(device)
        return self

    def cpu(self):
        self._F = self._F.cpu()
        return self

    def get_device(self):
        return self._F.get_device()

    def getKey(self):
        return self.coords_key
