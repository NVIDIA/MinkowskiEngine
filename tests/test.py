import numpy as np

import torch
from torch.autograd import gradcheck

import SparseConvolutionEngineFFI as SCE
from SparseConvolution import SparseConvolution
from Common import Metadata, RegionType, convert_to_long_tensor

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

    pixel_dist, stride, kernel_size, dilation, D = 1, 2, 2, 1, 2
    in_nchannel, out_nchannel = 2, 2
    coords = torch.from_numpy(np.array(coords)).long()
    in_feat = torch.FloatTensor(coords.size(0), in_nchannel).zero_()
    in_feat[1] = 1
    in_feat[2] = 2
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

    # conv.kernel.data[:] = torch.arange(9)
    # conv.bias.data[:] = torch.arange(out_nchannel) + 1
    print(conv.kernel.data.squeeze())
    print(in_feat)
    in_feat.requires_grad_()

    # The coords get initialized after the forward pass
    print(SCE.get_coords(coords2, pixel_dist * stride, D, metadata.ffi))
    out = conv(in_feat)
    print(SCE.get_coords(coords2, pixel_dist * stride, D, metadata.ffi))
    print(coords2)

    print(out.data.squeeze())

    # Permutation
    perm = torch.LongTensor()
    stride = convert_to_long_tensor(stride, D)
    SCE.get_permutation(perm, stride, pixel_dist, D, metadata.ffi)
    print(perm)

    grad = torch.zeros(out.size())
    grad[0] = 0.2
    grad[1] = 1
    out.backward(grad)
    print(in_feat.grad)

    print(
        gradcheck(
            conv, (in_feat, ),
            atol=1e-3,
            rtol=1e-2,
            eps=1e-4))

    # GPU
    if use_gpu:
        conv = conv.to(device)
        in_feat_cu = in_feat.to(device)
        out = conv(in_feat_cu)
        print(out)

        grad = grad.to(device)
        out.backward(grad)
        print(in_feat_cu.grad)

        print(
            gradcheck(
                conv, (in_feat_cu, ),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    metadata.clear()
