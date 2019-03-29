import torch
import unittest

from MinkowskiEngine import SparseTensor, MinkowskiPruning, MinkowskiPruningFunction

from utils.gradcheck import gradcheck
from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_pruning(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        use_feat = torch.rand(feats.size(0)) < 0.5
        pruning = MinkowskiPruning(D)
        output = pruning(input, use_feat)
        print(use_feat, output)

        # Check backward
        fn = MinkowskiPruningFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, use_feat, input.coords_key, output.coords_key,
                     input.C)))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            output = pruning(input, use_feat)
            print(output)

        self.assertTrue(
            gradcheck(
                fn, (input.F, use_feat, input.coords_key, output.coords_key,
                     input.C)))


if __name__ == '__main__':
    unittest.main()
