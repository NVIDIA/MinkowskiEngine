import unittest
from MinkowskiEngine import SparseTensor, SparseConvolution, \
    SparseMaxPoolingFunction, SparseMaxPooling, \
    SparseSumPoolingFunction, SparseSumPooling, \
    SparseNonzeroAvgPoolingFunction, SparseNonzeroAvgPooling, \
    SparseNonzeroAvgUnpoolingFunction, SparseNonzeroAvgUnpooling, \
    SparseGlobalAvgPoolingFunction, SparseGlobalAvgPooling
from gradcheck import gradcheck

from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_maxpool(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = SparseMaxPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = SparseMaxPoolingFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type, None, None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    def test_sumpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = SparseSumPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = SparseSumPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type, None, None, None, input.m),
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
