import torch
import unittest

from MinkowskiEngine import SparseTensor, MinkowskiConvolution, \
    MinkowskiSumPooling, \
    MinkowskiAvgPoolingFunction, MinkowskiAvgPooling, \
    MinkowskiPoolingTransposeFunction, MinkowskiPoolingTranspose, \
    MinkowskiGlobalPoolingFunction, MinkowskiGlobalPooling, \
    MinkowskiMaxPoolingFunction, MinkowskiMaxPooling
from gradcheck import gradcheck

from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_maxpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiMaxPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                     pool.dilation, pool.region_type, None, input.coords_key,
                     None, input.C)))

        if not torch.cuda.is_available():
          return

        device = torch.device('cuda')
        input = input.to(device)
        output = pool(input)
        print(output)

        # Check backward
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                     pool.dilation, pool.region_type, None, input.coords_key,
                     None, input.C)))

    def test_sumpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiSumPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                     pool.dilation, pool.region_type, None, False,
                     input.coords_key, None, input.C)))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            pool = pool.to(device)
            output = pool(input)
            print(output)

    def test_avgpooling_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            pool = pool.to(device)
            output = pool(input)
            print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                     pool.dilation, pool.region_type, None, True,
                     input.coords_key, None, input.C)))

    def test_avgpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, pool.stride, pool.kernel_size,
                     pool.dilation, pool.region_type, None, True,
                     input.coords_key, None, input.C)))

    def test_global_avgpool(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiGlobalPooling(dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiGlobalPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, 0, True, input.coords_key, None, input.C)))

    def test_unpool(self):
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        conv = conv.double()
        unpool = MinkowskiPoolingTranspose(kernel_size=3, stride=2, dimension=D)
        input = conv(input)
        output = unpool(input)
        print(output)

        # Check backward
        fn = MinkowskiPoolingTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, unpool.stride,
                     unpool.kernel_size, unpool.dilation, unpool.region_type,
                     None, False, input.coords_key, None, input.C)))

    def test_unpooling_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        conv = conv.double()
        unpool = MinkowskiPoolingTranspose(kernel_size=3, stride=2, dimension=D)
        input = conv(input)
        output = unpool(input)
        print(output)

        # Check backward
        fn = MinkowskiPoolingTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, unpool.stride,
                     unpool.kernel_size, unpool.dilation, unpool.region_type,
                     None, False, input.coords_key, None, input.C)))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            output = unpool(input)
            print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.pixel_dist, unpool.stride,
                     unpool.kernel_size, unpool.dilation, unpool.region_type,
                     None, True, input.coords_key, None, input.C)))


if __name__ == '__main__':
    unittest.main()
