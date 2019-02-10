import torch
import torch.nn as nn
from torch.nn import Module

from SparseTensor import SparseTensor
from MinkowskiPooling import MinkowskiGlobalPooling
from MinkowskiBroadcast import MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication


class MinkowskiBatchNorm(Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(MinkowskiBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)

    def forward(self, input):
        output = self.bn(input.F)
        return SparseTensor(
            output, coords_key=input.coords_key, coords_manager=input.C)


class MinkowskiInstanceNorm(Module):

    def __init__(self, num_features, eps=1e-5, D=-1):
        super(MinkowskiInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

        self.mean_in = MinkowskiGlobalPooling(dimension=D)
        self.glob_sum = MinkowskiBroadcastAddition(dimension=D)
        self.glob_sum2 = MinkowskiBroadcastAddition(dimension=D)
        self.glob_mean = MinkowskiGlobalPooling(dimension=D)
        self.glob_times = MinkowskiBroadcastMultiplication(dimension=D)
        self.D = D
        self.reset_parameters()

    def __repr__(self):
        s = f'(nchannels={self.num_features}, D={self.D})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        neg_mean_in = self.mean_in(
            SparseTensor(-x.F, coords_key=x.coords_key, coords_manager=x.C))
        centered_in = self.glob_sum(x, neg_mean_in)
        temp = SparseTensor(
            centered_in.F**2,
            coords_key=centered_in.coords_key,
            coords_manager=centered_in.C)
        var_in = self.glob_mean(temp)
        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coords_key=var_in.coords_key,
            coords_manager=var_in.C)

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(
            x.F * self.weight + self.bias,
            coords_key=x.coords_key,
            coords_manager=x.C)
