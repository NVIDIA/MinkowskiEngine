import torch
from torch.nn import Module

from SparseTensor import SparseTensor


class SparseBatchNorm(Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(SparseBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)

    def forward(self, input):
        output = self.bn(input.F)
        return SparseTensor(
            output,
            coords=input.C,
            coords_key=input.coords_key,
            pixel_dist=input.pixel_dist,
            net_metadata=input.m)
