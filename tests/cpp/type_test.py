import unittest
import torch
import MinkowskiEngineTest._C


class TypeTestCase(unittest.TestCase):

    def test(self):
        MinkowskiEngineTest._C.type_test()
        self.assertErtTrue(True)
