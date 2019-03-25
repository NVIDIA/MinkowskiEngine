import torch
import unittest

from gradcheck import gradcheck

from MinkowskiEngine import SparseTensor, MinkowskiAdaptiveDilationConvolution, \
    MinkowskiAdaptiveDilationConvolutionFunction, initialize_nthreads

from tests.common import data_loader


class TestConvolution(unittest.TestCase):

    def test(self):
        if not torch.cuda.is_available():
            return
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        dilations_ = ((torch.rand(len(input), D) > 0.5) + 1).int()
        dilations = SparseTensor(dilations_, coords=coords)
        # Initialize context
        conv = MinkowskiAdaptiveDilationConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            has_bias=True,
            dimension=D)
        conv = conv.double()
        output = conv(input, dilations)
        print(output)

        kernel_map = input.C.get_kernel_map(1, 1, stride=1, kernel_size=3)
        print(kernel_map)
        print(dilations)

        # Check backward
        fn = MinkowskiAdaptiveDilationConvolutionFunction()

        grad = output.F.clone().zero_()
        grad[0] = 1
        output.F.backward(grad)

        self.assertTrue(
            gradcheck(
                fn,
                (input.F, conv.kernel, dilations.F, input.pixel_dist,
                 conv.stride, conv.kernel_size, conv.dilation, conv.region_type,
                 None, input.coords_key, None, input.coords_man)))


if __name__ == '__main__':
    initialize_nthreads(3, D=2)
    unittest.main()
