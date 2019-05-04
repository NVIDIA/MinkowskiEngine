import torch
import unittest

from MinkowskiEngine import SparseTensor, MinkowskiGlobalPooling, \
    MinkowskiBroadcastFunction, MinkowskiBroadcastAddition, \
    MinkowskiBroadcastMultiplication, OperationType

from utils.gradcheck import gradcheck
from tests.common import data_loader


class TestBroadcast(unittest.TestCase):

    def test_broadcast_gpu(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        coords, feats_glob, labels = data_loader(in_channels)
        feats = feats.double()
        feats_glob = feats_glob.double()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiGlobalPooling(dimension=D)
        input_glob = pool(input)
        input_glob.F.requires_grad_()
        broadcast = MinkowskiBroadcastAddition(D)
        output = broadcast(input, input_glob)
        print(output)

        # Check backward
        fn = MinkowskiBroadcastFunction()

        device = torch.device('cuda')
        input = input.to(device)
        input_glob = input_glob.to(device)
        output = broadcast(input, input_glob)
        print(output)
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input_glob.F, OperationType.ADDITION,
                 input.coords_key, input_glob.coords_key, input.coords_man)))

        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input_glob.F, OperationType.MULTIPLICATION,
                 input.coords_key, input_glob.coords_key, input.coords_man)))

    def test_broadcast(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        coords, feats_glob, labels = data_loader(in_channels)
        feats = feats.double()
        feats_glob = feats_glob.double()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiGlobalPooling(dimension=D)
        input_glob = pool(input)
        input_glob.F.requires_grad_()
        broadcast = MinkowskiBroadcastAddition(D)
        broadcast_mul = MinkowskiBroadcastMultiplication(D)
        output = broadcast(input, input_glob)
        print(output)
        output = broadcast_mul(input, input_glob)
        print(output)

        # Check backward
        fn = MinkowskiBroadcastFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input_glob.F, OperationType.ADDITION,
                 input.coords_key, input_glob.coords_key, input.coords_man)))

        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input_glob.F, OperationType.MULTIPLICATION,
                 input.coords_key, input_glob.coords_key, input.coords_man)))


if __name__ == '__main__':
    unittest.main()
