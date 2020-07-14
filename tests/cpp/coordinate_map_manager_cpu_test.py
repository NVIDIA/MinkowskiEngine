import unittest
import torch
import MinkowskiEngineTest._C as _C


class CoordinateMapManagerTestCase(unittest.TestCase):
    def test(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        key, manager, map_inverse_map = _C.coordinate_map_manager_test(coordinates, "")
        unique_map, inverse_map = map_inverse_map
        self.assertTrue(
            torch.all(coordinates[unique_map.long()][inverse_map.long()] == coordinates)
        )

    def test_stride(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        key, manager, map_inverse_map = _C.coordinate_map_manager_test(coordinates, "")
        unique_map, inverse_map = map_inverse_map
        stride = [2]
        key = _C.coordinate_map_manager_stride(manager, key, stride)
        print(key)

    def test_kernel_map(self):
        coordinates = torch.IntTensor([[0, 1], [0, 2], [1, 2], [1, 3]])
        manager = _C.CoordinateMapManager()
        key, (unique_map, inverse_map) = manager.insert_and_map(coordinates, [1], "1")
        key2, (unique_map2, inverse_map2) = manager.insert_and_map(
            coordinates, [1], "2"
        )
        print(key, key2)
        self.assertTrue(
            torch.all(coordinates[unique_map.long()][inverse_map.long()] == coordinates)
        )
        in_maps, out_maps = _C.coordinate_map_manager_kernel_map(
            manager, key, key2, [3]
        )

        print(in_maps)
        print(out_maps)
