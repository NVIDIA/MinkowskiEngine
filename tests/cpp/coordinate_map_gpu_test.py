import unittest
import torch
import MinkowskiEngineTest._C


class CoordinateMapTestCase(unittest.TestCase):
    def test_batch_insert(self):
        assert torch.cuda.is_available()
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]]).to(0)
        self.assertEqual(
            MinkowskiEngineTest._C.coordinate_map_batch_insert_test(coordinates), 3
        )

    def test_batch_find(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]]).to(0)
        queries = torch.IntTensor([[-1, 1], [1, 2], [2, 3], [2, 3], [0, 0]]).to(0)
        valid_query_index, query_value = MinkowskiEngineTest._C.coordinate_map_batch_find_test(
            coordinates, queries
        )
        self.assertEqual(len(valid_query_index), len(query_value))
        self.assertEqual(len(valid_query_index), 3)

        self.assertEqual(valid_query_index[0], 1)
        self.assertEqual(valid_query_index[1], 2)
        self.assertEqual(valid_query_index[2], 3)

        self.assertEqual(query_value[0], 1)
        self.assertEqual(query_value[1], 2)
        self.assertEqual(query_value[2], 2)
