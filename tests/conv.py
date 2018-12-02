import unittest
from MinkowskiEngine import SparseTensor, RegionType, \
    SparseConvolution, SparseConvolutionFunction, \
    SparseConvolutionTranspose, SparseConvolutionTransposeFunction
from gradcheck import gradcheck

from tests.common import data_loader


class TestConvolution(unittest.TestCase):

    def test(self):
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        conv = SparseConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        output = conv(input)
        print(output)

        # Check backward
        fn = SparseConvolutionFunction()

        self.assertTrue(
            gradcheck(
                fn, (input.F, conv.kernel, input.pixel_dist, conv.stride,
                     conv.kernel_size, conv.dilation, conv.region_type, None,
                     None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    def test_hybrid(self):
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        conv = SparseConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            region_type=RegionType.HYBRID,
            axis_types=[RegionType.HYPERCROSS, RegionType.HYPERCROSS],
            dimension=D)
        output = conv(input)
        print(output)

        # Check backward
        fn = SparseConvolutionFunction()

        self.assertTrue(
            gradcheck(
                fn, (input.F, conv.kernel, input.pixel_dist, conv.stride,
                     conv.kernel_size, conv.dilation, conv.region_type_,
                     conv.region_offset_, None, None, input.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))


class TestConvolutionTranspose(unittest.TestCase):

    def test(self):
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        input = SparseTensor(feats, coords=coords)
        conv = SparseConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        conv_tr = SparseConvolutionTranspose(
            out_channels,
            in_channels,
            kernel_size=3,
            upsample_stride=2,
            dimension=D)

        input_tr = conv(input)
        output = conv_tr(input_tr)
        print(output)

        # Check backward
        fn = SparseConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn, (input_tr.F, conv_tr.kernel, input_tr.pixel_dist,
                     conv_tr.stride, conv_tr.kernel_size, conv_tr.dilation,
                     conv_tr.region_type, None, None, None, input_tr.m),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))


if __name__ == '__main__':
    unittest.main()
