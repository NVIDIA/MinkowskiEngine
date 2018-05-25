import numpy as np

import torch
import torch.nn as nn
from gradcheck import gradcheck

import SparseConvolutionEngineFFI as SCE
from SparseConvolution import SparseConvolution, SparseConvolutionTranspose
from Common import Metadata, RegionType, convert_to_long_tensor


class ConvDeconv(nn.Module):
    def __init__(self, in_nchannel, out_nchannel, D, metadata):
        super(ConvDeconv, self).__init__()
        self.conv = SparseConvolution(
            in_nchannel,
            out_nchannel,
            pixel_dist=1,
            kernel_size=3,
            stride=2,
            dilation=1,
            region_type=RegionType.HYPERCUBE,
            has_bias=True,
            dimension=D,
            metadata=metadata)

        # Conv transpose
        self.conv_tr = SparseConvolutionTranspose(
            in_channels=out_nchannel,
            out_channels=in_nchannel,
            pixel_dist=2,
            kernel_size=3,
            upsample_stride=2,
            dilation=1,
            region_type=RegionType.HYPERCUBE,
            region_offset=None,
            has_bias=True,
            dimension=D,
            metadata=metadata)

        self.D = D
        self.metadata = metadata

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_tr(out)
        return out

    def initialize_coords(self, coords, pixel_dist):
        SCE.initialize_coords(coords, pixel_dist, self.D, self.metadata.ffi)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_gpu = True
    IN1 = [" X  ", "XXX ", "    "]
    IN2 = ["X   ", "  XX", "X  X"]
    coords = []
    for i, row in enumerate(IN1):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j, 0])  # Last element for batch index
    for i, row in enumerate(IN2):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j, 1])  # Last element for batch index

    pixel_dist, stride, kernel_size, dilation, D = 1, 2, 3, 1, 2
    in_nchannel, out_nchannel = 2, 2
    coords = torch.from_numpy(np.array(coords)).long()
    in_feat = torch.FloatTensor(coords.size(0), in_nchannel).normal_()
    # import ipdb; ipdb.set_trace()
    # in_feat[1] = 1
    # in_feat[2] = 2
    # in_feat[4] = 5
    # in_feat[8] = 7
    metadata = Metadata(D)

    pixel_dist = convert_to_long_tensor(pixel_dist, D)
    SCE.initialize_coords(coords, pixel_dist, D, metadata.ffi)

    coords2 = torch.LongTensor()
    print(SCE.get_coords(coords2, pixel_dist, D, metadata.ffi))
    print(coords2)

    conv = SparseConvolution(
        in_nchannel,
        out_nchannel,
        pixel_dist,
        kernel_size,
        stride,
        dilation,
        region_type=RegionType.HYPERCUBE,
        has_bias=True,
        dimension=D,
        metadata=metadata)

    print(conv.kernel.data.squeeze())
    print(in_feat)
    in_feat.requires_grad_()

    # The coords get initialized after the forward pass
    coords3 = torch.LongTensor()
    print(SCE.get_coords(coords3, pixel_dist * stride, D, metadata.ffi))
    out = conv(in_feat)
    print(SCE.get_coords(coords3, pixel_dist * stride, D, metadata.ffi))
    print(coords3)

    print(out.squeeze())

    # Permutation
    perm = torch.LongTensor()
    SCE.get_permutation(perm, pixel_dist * stride, pixel_dist, D, metadata.ffi)
    print(perm)

    # Conv transpose
    conv_tr = SparseConvolutionTranspose(
        in_channels=out_nchannel,
        out_channels=in_nchannel,
        pixel_dist=pixel_dist * stride,
        kernel_size=2,
        upsample_stride=stride,
        dilation=dilation,
        region_type=RegionType.HYPERCUBE,
        region_offset=None,
        has_bias=True,
        dimension=D,
        metadata=metadata)

    # conv_tr.kernel.data[:] = torch.from_numpy(
    #     np.random.permutation(kernel_size**D * in_nchannel * out_nchannel))
    print(conv_tr.kernel.squeeze())
    tr_out = conv_tr(out)
    print(tr_out)

    grad = torch.zeros(out.size())
    grad[0] = 0.2
    grad[1] = 1
    # grad[1, 0] = 1
    # grad[1, 1] = - 1
    # grad[0, 1] = 0.2
    out.backward(grad)
    print(in_feat.grad)

    tr_grad = torch.zeros_like(tr_out)
    tr_grad[0] = 0.2
    tr_grad[1] = 1
    tr_out.backward(tr_grad)
    print(out.grad)

    print(gradcheck(
        conv, (in_feat, ),
        atol=1e-3,
        rtol=1e-2,
        eps=1e-4))

    print(gradcheck(
        conv_tr, (out, ),
        atol=1e-3,
        rtol=1e-2,
        eps=1e-4))

    model = ConvDeconv(in_nchannel, out_nchannel, D, Metadata(D))
    model.initialize_coords(coords, pixel_dist)
    out = model(in_feat)
    print(out)
    print(gradcheck(
        model, (in_feat, ),
        atol=1e-3,
        rtol=1e-2,
        eps=1e-4))

    # GPU
    if use_gpu:
        conv = conv.to(device)
        in_feat_cu = in_feat.to(device)
        out = conv(in_feat_cu)
        print(out)

        conv_tr = conv_tr.to(device)
        tr_out = conv_tr(out)
        print(tr_out)

        grad = grad.to(device)
        out.backward(grad)
        print(in_feat_cu.grad)

        model = model.to(device)

        print(gradcheck(
            conv, (in_feat_cu, ),
            atol=1e-3,
            rtol=1e-2,
            eps=1e-4))

        print(gradcheck(
            conv_tr, (out, ),
            atol=1e-3,
            rtol=1e-2,
            eps=1e-4))

        print(gradcheck(
            model, (in_feat_cu, ),
            atol=1e-3,
            rtol=1e-2,
            eps=1e-4))
    metadata.clear()
