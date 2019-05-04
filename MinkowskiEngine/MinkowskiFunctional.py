import torch.nn.functional as F

from SparseTensor import SparseTensor


def relu(input):
    output = F.relu(input.F)
    return SparseTensor(
        output, coords_key=input.coords_key, coords_manager=input.coords_man)
