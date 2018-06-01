import numpy as np

import torch
from gradcheck import gradcheck

import SparseConvolutionEngineFFI as SCE
from Common import Metadata, RegionType, convert_to_int_tensor
from SparsePooling import SparseMaxPooling

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
    # for i, row in enumerate(IN2):
    #     for j, col in enumerate(row):
    #         if col != ' ':
    #             coords.append([i, j, 1])  # Last element for batch index

    pixel_dist, stride, kernel_size, dilation, D = 1, 2, 2, 1, 2
    in_nchannel = 2
    coords = torch.from_numpy(np.array(coords)).int()
    in_feat = torch.FloatTensor(coords.size(0), in_nchannel).normal_()
    # .zero_()
    # in_feat[1, 0] = 1
    # in_feat[1, 2] = 1
    # in_feat[2] = 2
    metadata = Metadata(D)

    pixel_dist = convert_to_int_tensor(pixel_dist, D)
    SCE.initialize_coords(coords, pixel_dist, D, metadata.ffi)

    coords2 = torch.IntTensor()
    print(SCE.get_coords(coords2, pixel_dist, D, metadata.ffi))
    print(coords2)
    print(in_feat)

    pooling = SparseMaxPooling(
        pixel_dist,
        kernel_size,
        stride,
        dilation,
        region_type=RegionType.HYPERCUBE,
        dimension=D,
        metadata=metadata)

    in_feat.requires_grad_()

    # The coords get initialized after the forward pass
    print(SCE.get_coords(coords2, pixel_dist * stride, D, metadata.ffi))
    out = pooling(in_feat)
    print(SCE.get_coords(coords2, pixel_dist * stride, D, metadata.ffi))
    print(coords2)

    print(out.data.squeeze())

    # Permutation
    perm = torch.IntTensor()
    SCE.get_permutation(perm, pixel_dist * stride, pixel_dist, D, metadata.ffi)
    print(perm)

    grad = torch.zeros(out.size())
    grad[1] = 1
    # grad[1, 1] = 3
    # # grad[1, 0] = 1
    # # grad[1, 1] = - 1
    # # grad[0, 1] = 0.2
    out.backward(grad)
    print(in_feat.grad)

    print(
        gradcheck(
            pooling, (in_feat, ),
            atol=1e-3,
            rtol=1e-2,
            eps=1e-4))

    # GPU
    if use_gpu:
        pooling = pooling.to(device)
        in_feat_cu = in_feat.to(device)
        print(in_feat_cu)
        out = pooling(in_feat_cu)
        print(out)

        grad = grad.cuda()
        out.backward(grad)
        print(in_feat_cu.grad)

        print(
            gradcheck(
                pooling, (in_feat_cu, ),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    metadata.clear()
