import torch
import unittest

from MinkowskiEngine import SparseTensor, MinkowskiPruning, MinkowskiPruningFunction
from gradcheck import gradcheck

from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_sumpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
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
                     input.C),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            output = pruning(input, use_feat)
            print(output)


if __name__ == '__main__':
    unittest.main()
