import numpy as np

import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

import SparseConvolutionEngineFFI as SCE
from Common import Metadata, convert_to_int_tensor
from SparsePooling import SparseGlobalAvgPooling
from SparseBroadcast import SparseGlobalBroadcastAddition

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
        return hook


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
    for i, row in enumerate(IN1):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j, 2])  # Last element for batch index

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

    pooling = SparseGlobalAvgPooling(
        pixel_dist,
        batch_size=0,
        dimension=D,
        metadata=metadata)

    # The coords get initialized after the forward pass
    print(SCE.get_coords(coords2, pixel_dist * 0, D, metadata.ffi))
    in_feat_glob = pooling(in_feat)
    print(in_feat_glob)
    print(SCE.get_coords(coords2, pixel_dist * 0, D, metadata.ffi))
    print(coords2)

    broadcast_addition = SparseGlobalBroadcastAddition(pixel_dist, D, metadata)

    in_feat.requires_grad_()
    in_feat_glob.requires_grad_()

    out_t = broadcast_addition(in_feat, in_feat_glob)
    print(out_t)

    grad = torch.zeros(out_t.size())
    grad[6] = 1
    # grad[1, 1] = 3
    # # grad[1, 0] = 1
    # # grad[1, 1] = - 1
    # # grad[0, 1] = 0.2
    out_t.backward(grad)
    print(in_feat_glob.grad)

    print(
        gradcheck(
            broadcast_addition, (in_feat, in_feat_glob),
            atol=1e-3,
            rtol=1e-2,
            eps=1e-4))

    # GPU
    if use_gpu:
        broadcast_addition = broadcast_addition.to(device)
        in_feat_cu = Variable(in_feat.to(device), requires_grad=True)
        in_feat_glob_cu = Variable(in_feat_glob.to(device), requires_grad=True)
        print(in_feat_cu)
        out = broadcast_addition(in_feat_cu, in_feat_glob_cu)
        print(out)

        grad = grad.cuda()

        out.backward(grad)
        print(in_feat_cu.grad)
        print(in_feat_glob_cu.grad)

        print(
            gradcheck(
                broadcast_addition, (in_feat_cu, in_feat_glob_cu),
                atol=1e-3,
                rtol=1e-2,
                eps=1e-4))

    metadata.clear()
