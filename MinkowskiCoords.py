import numpy as np
import torch
from Common import convert_to_int_list
import MinkowskiEngineBackend as MEB


class CoordsKey():

    def __init__(self, D):
        self.D = D
        self.CPPCoordsKey = getattr(MEB, f'PyCoordsKey{self.D}')()

    def setKey(self, key):
        self.CPPCoordsKey.setKey(key)

    def getKey(self):
        return self.CPPCoordsKey.getKey()

    def setPixelDist(self, pixel_dist):
        pixel_dist = convert_to_int_list(pixel_dist, self.D)
        self.CPPCoordsKey.setPixelDist(pixel_dist)

    def getPixelDist(self):
        return self.CPPCoordsKey.getPixelDist()

    def __repr__(self):
        return str(self.CPPCoordsKey)


class CoordsManager():

    def __init__(self, D):
        self.D = D
        self.CPPCoordsManager = getattr(MEB, f'PyCoordsManager{D}int32')()

    def initialize(self, coords, coords_key, enforce_creation=False):
        assert isinstance(coords_key, CoordsKey)
        self.CPPCoordsManager.initializeCoords(coords, coords_key.CPPCoordsKey,
                                               enforce_creation)

    def initialize_enforce(self, coords, coords_key):
        assert isinstance(coords_key, CoordsKey)
        self.CPPCoordsManager.initializeCoords(coords, coords_key.CPPCoordsKey,
                                               True)

    def get_coords_key(self, pixel_dists):
        pixel_dists = convert_to_int_list(pixel_dists, self.D)
        key = self.CPPCoordsManager.getCoordsKey(pixel_dists)
        coords_key = CoordsKey(self.D)
        coords_key.setKey(key)
        coords_key.setPixelDist(pixel_dists)
        return coords_key

    def get_coords(self, coords_key):
        assert isinstance(coords_key, CoordsKey)
        coords = torch.IntTensor()
        self.CPPCoordsManager.getCoords(coords, coords_key.CPPCoordsKey)
        return coords

    def get_coords_size_by_coords_key(self, coords_key):
        assert isinstance(coords_key, CoordsKey)
        return self.CPPCoordsManager.getCoordsSize(coords_key.CPPCoordsKey)

    def get_mapping_by_pixel_dists(self, in_pixel_dists, out_pixel_dists):
        in_key = self.get_coords_key(in_pixel_dists)
        out_key = self.get_coords_key(out_pixel_dists)
        return self.get_mapping_by_coords_key(in_key, out_key)

    def get_mapping_by_coords_key(self, in_coords_key, out_coords_key):
        assert isinstance(in_coords_key, CoordsKey) \
            and isinstance(out_coords_key, CoordsKey)
        mapping = torch.IntTensor()
        self.CPPCoordsManager.getCoordsMapping(
            mapping, in_coords_key.CPPCoordsKey, out_coords_key.CPPCoordsKey)
        return mapping

    def permute_label(self,
                      label,
                      max_label,
                      target_pixel_dist,
                      label_pixel_dist=1):
        """
        """
        if target_pixel_dist == label_pixel_dist:
            return label

        label_coords_key = self.get_coords_key(label_pixel_dist)
        target_coords_key = self.get_coords_key(target_pixel_dist)

        permutation = self.get_mapping_by_coords_key(label_coords_key,
                                                     target_coords_key)
        nrows = self.get_coords_size_by_coords_key(target_coords_key)

        label = label.contiguous().numpy()
        permutation = permutation.numpy()

        counter = np.zeros((nrows, max_label), dtype='int32')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def __repr__(self):
        return str(self.CPPCoordsManager)
