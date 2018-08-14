import numpy as np

import torch
from torch.autograd import gradcheck

import SparseConvolutionEngineFFI as SCE
from SparseConvolution import SparseConvolution
from Common import NetMetadata, RegionType, convert_to_int_tensor

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
    coords = torch.from_numpy(np.array(coords)).int()
    in_feat = torch.FloatTensor(coords.size(0), in_nchannel).zero_()
    in_feat[1] = 1
    in_feat[2] = 2
    net_metadata = NetMetadata(D)

    pixel_dist = convert_to_int_tensor(pixel_dist, D)
    SCE.initialize_coords(coords, pixel_dist, D, net_metadata.ffi)

    coords2 = torch.IntTensor()
    print(SCE.get_coords(coords2, pixel_dist, D, net_metadata.ffi))
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
        net_metadata=net_metadata)

    # conv.kernel.data[:] = torch.arange(9)
    # conv.bias.data[:] = torch.arange(out_nchannel) + 1
    print(conv.kernel.data.squeeze())
    print(in_feat)
    in_feat.requires_grad_()

    # The coords get initialized after the forward pass
    print(SCE.get_coords(coords2, pixel_dist * stride, D, net_metadata.ffi))
    out = conv(in_feat)
    print(SCE.get_coords(coords2, pixel_dist * stride, D, net_metadata.ffi))
    print(coords2)

    print(out.data.squeeze())

    # Permutation
    perm = torch.IntTensor()
    stride = convert_to_int_tensor(stride, D)
    SCE.get_permutation(perm, stride, pixel_dist, D, net_metadata.ffi)
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

    net_metadata.clear()
