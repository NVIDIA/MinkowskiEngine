import unittest
import torch
import MinkowskiEngineTest._C


class CoordinateMapTestCase(unittest.TestCase):
    def test_batch_insert(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        self.assertEqual(
            MinkowskiEngineTest._C.coordinate_map_batch_insert_test(coordinates), 3
        )

    def test_batch_find(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        queries = torch.IntTensor([[-1, 1], [1, 2], [2, 3], [2, 3], [0, 0]])
        (
            valid_query_index,
            query_value,
        ) = MinkowskiEngineTest._C.coordinate_map_batch_find_test(coordinates, queries)
        self.assertEqual(len(valid_query_index), len(query_value))
        self.assertEqual(len(valid_query_index), 3)

        self.assertEqual(valid_query_index[0], 1)
        self.assertEqual(valid_query_index[1], 2)
        self.assertEqual(valid_query_index[2], 3)

        self.assertEqual(query_value[0], 1)
        self.assertEqual(query_value[1], 2)
        self.assertEqual(query_value[2], 2)

    def test_stride(self):
        coordinates = torch.IntTensor([[0, 1], [0, 2], [0, 3], [0, 3]])
        stride = [1]
        with self.assertRaises(TypeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        stride = torch.IntTensor([-1])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        stride = torch.IntTensor([1, 1])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        stride = torch.IntTensor([2])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [2])

        coordinates = torch.IntTensor(
            [[0, 1, 1], [0, 2, 1], [0, 1, 0], [1, 0, 3], [1, 0, 2]]
        )
        stride = torch.IntTensor([1])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        coordinates = torch.IntTensor(
            [[0, 1, 1], [0, 2, 1], [0, 1, 0], [1, 0, 3], [1, 0, 2]]
        )
        stride = torch.IntTensor([1, 1])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 5)
        self.assertEqual(tensor_stride, [1, 1])

        stride = torch.IntTensor([2, 1])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 5)
        self.assertEqual(tensor_stride, [2, 1])

        stride = torch.IntTensor([4, 4])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [4, 4])

        coordinates = torch.IntTensor(
            [[0, -1], [0, -2], [0, 1], [0, 0]]
        )
        stride = torch.IntTensor([2])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [2])
