import torch
from SparseTensor import SparseTensor


def cat(a, b):
    assert isinstance(a, SparseTensor) and isinstance(b, SparseTensor)
    assert a.m == b.m
    assert a.coords_key[0] == b.coords_key[0]
    return SparseTensor(
        torch.cat((a.F, b.F), dim=1),
        pixel_dist=a.pixel_dist,
        coords=a.C,
        coords_key=a.coords_key,
        net_metadata=a.m)
