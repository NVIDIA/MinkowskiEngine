import torch
import unittest

from MinkowskiEngine import SparseTensor, MinkowskiInstanceNorm, MinkowskiInstanceNormFunction
from utils.gradcheck import gradcheck

from tests.common import data_loader


class TestNormalization(unittest.TestCase):

    def test_inst_norm(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        input.F.requires_grad_()
        norm = MinkowskiInstanceNorm(
            num_features=in_channels, dimension=D).double()

        out = norm(input)
        print(out)

        fn = MinkowskiInstanceNormFunction()
        self.assertTrue(
            gradcheck(fn,
                      (input.F, 0, input.coords_key, None, input.coords_man)))

    def test_inst_norm_gpu(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()

        device = torch.device('cuda')
        input = SparseTensor(feats, coords=coords).to(device)
        input.F.requires_grad_()
        norm = MinkowskiInstanceNorm(
            num_features=in_channels, dimension=D).to(device).double()

        out = norm(input)
        print(out)

        fn = MinkowskiInstanceNormFunction()
        self.assertTrue(
            gradcheck(fn,
                      (input.F, 0, input.coords_key, None, input.coords_man)))


if __name__ == '__main__':
    unittest.main()
