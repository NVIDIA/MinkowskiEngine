import unittest
from MinkowskiEngine import OperationType, SparseTensor, SparseConvolution, \
    SparseNonzeroAvgPoolingFunction, SparseNonzeroAvgPooling, \
    SparseNonzeroAvgUnpoolingFunction, SparseNonzeroAvgUnpooling, \
    SparseGlobalAvgPoolingFunction, SparseGlobalAvgPooling, \
    SparseGlobalBroadcastAddition, SparseGlobalBroadcastFunction
from gradcheck import gradcheck

from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_broadcast(self):
        in_channels, D = 1, 2
        coords, feats, labels = data_loader(in_channels)
        input = SparseTensor(feats, coords=coords)
        pool = SparseGlobalAvgPooling(dimension=D)
        input_glob = pool(input)

        input_glob.F.requires_grad_()
        broadcast = SparseGlobalBroadcastAddition(D)
        output = broadcast(input, input_glob)
        print(output)

        # Check backward
        fn = SparseGlobalBroadcastFunction()

        self.assertTrue(
            gradcheck(
                fn, (input.F, input_glob.F, input.pixel_dist,
                     OperationType.ADDITION, None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

        self.assertTrue(
            gradcheck(
                fn, (input.F, input_glob.F, input.pixel_dist,
                     OperationType.MULTIPLICATION, None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    def test_avgpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = SparseNonzeroAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = SparseNonzeroAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type, None, None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    def test_global_avgpool(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = SparseGlobalAvgPooling(dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = SparseGlobalAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, 0, None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    def test_unpool(self):
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        input = SparseTensor(feats, coords=coords)
        conv = SparseConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        unpool = SparseNonzeroAvgUnpooling(kernel_size=3, stride=2, dimension=D)
        input_tr = conv(input)
        output = unpool(input_tr)
        print(output)

        # Check backward
        fn = SparseNonzeroAvgUnpoolingFunction()

        self.assertTrue(
            gradcheck(
                fn, (input_tr.F, input_tr.pixel_dist, unpool.stride,
                     unpool.kernel_size, unpool.dilation, unpool.region_type,
                     None, None, None, input_tr.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))


if __name__ == '__main__':
    unittest.main()
