# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Function

from SparseTensor import SparseTensor
from MinkowskiPooling import MinkowskiGlobalPooling
from MinkowskiBroadcast import MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication, OperationType, operation_type_to_int
from MinkowskiCoords import CoordsKey
from Common import get_minkowski_function, GlobalPoolingMode


class MinkowskiBatchNorm(Module):
    r"""A batch normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
    """

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
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        s = '({}, eps={}, momentum={}, affine={}, track_running_stats={})'.format(
            self.bn.num_features, self.bn.eps, self.bn.momentum, self.bn.affine,
            self.bn.track_running_stats)
        return self.__class__.__name__ + s


class MinkowskiSyncBatchNorm(MinkowskiBatchNorm):
    r"""A batch normalization layer with multi GPU synchronization.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 process_group=None):
        Module.__init__(self)
        self.bn = torch.nn.SyncBatchNorm(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            process_group=process_group)

    def forward(self, input):
        output = self.bn(input.F)
        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert
        :attr:`MinkowskiEngine.MinkowskiBatchNorm` layer in the model to
        :attr:`MinkowskiEngine.MinkowskiSyncBatchNorm` layer.

        Args:
            module (nn.Module): containing module
            process_group (optional): process group to scope synchronization,
            default is the whole world

        Returns:
            The original module with the converted
            :attr:`MinkowskiEngine.MinkowskiSyncBatchNorm` layer

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100)
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, MinkowskiBatchNorm):
            module_output = MinkowskiSyncBatchNorm(
                module.bn.num_features, module.bn.eps, module.bn.momentum,
                module.bn.affine, module.bn.track_running_stats, process_group)
            if module.bn.affine:
                module_output.bn.weight.data = module.bn.weight.data.clone(
                ).detach()
                module_output.bn.bias.data = module.bn.bias.data.clone().detach(
                )
                # keep reuqires_grad unchanged
                module_output.bn.weight.requires_grad = module.bn.weight.requires_grad
                module_output.bn.bias.requires_grad = module.bn.bias.requires_grad
            module_output.bn.running_mean = module.bn.running_mean
            module_output.bn.running_var = module.bn.running_var
            module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class MinkowskiInstanceNormFunction(Function):

    @staticmethod
    def forward(ctx,
                in_feat,
                mode=GlobalPoolingMode.AUTO,
                in_coords_key=None,
                glob_coords_key=None,
                coords_manager=None):
        assert isinstance(mode, GlobalPoolingMode), \
            f"Mode must be an instance of GlobalPoolingMode, {mode}"
        if glob_coords_key is None:
            glob_coords_key = CoordsKey(in_coords_key.D)

        gpool_forward = get_minkowski_function('GlobalPoolingForward', in_feat)
        broadcast_forward = get_minkowski_function('BroadcastForward', in_feat)
        add = operation_type_to_int(OperationType.ADDITION)
        multiply = operation_type_to_int(OperationType.MULTIPLICATION)

        mean = in_feat.new()
        num_nonzero = in_feat.new()

        cpp_in_coords_key = in_coords_key.CPPCoordsKey
        cpp_glob_coords_key = glob_coords_key.CPPCoordsKey
        cpp_coords_manager = coords_manager.CPPCoordsManager

        mean, num_nonzero = gpool_forward(in_feat, cpp_in_coords_key,
                                          cpp_glob_coords_key,
                                          cpp_coords_manager, True, mode.value)
        # X - \mu
        centered_feat = broadcast_forward(in_feat, -mean, add,
                                          cpp_in_coords_key,
                                          cpp_glob_coords_key,
                                          cpp_coords_manager)

        # Variance = 1/N \sum (X - \mu) ** 2
        variance, num_nonzero = gpool_forward(centered_feat**2,
                                              cpp_in_coords_key,
                                              cpp_glob_coords_key,
                                              cpp_coords_manager, True,
                                              mode.value)

        # norm_feat = (X - \mu) / \sigma
        inv_std = 1 / (variance + 1e-8).sqrt()
        norm_feat = broadcast_forward(centered_feat, inv_std, multiply,
                                      cpp_in_coords_key, cpp_glob_coords_key,
                                      cpp_coords_manager)

        ctx.mode = mode
        ctx.in_coords_key, ctx.glob_coords_key = in_coords_key, glob_coords_key
        ctx.coords_manager = coords_manager
        # For GPU tensors, must use save_for_backward.
        ctx.save_for_backward(inv_std, norm_feat)
        return norm_feat

    @staticmethod
    def backward(ctx, out_grad):
        # https://kevinzakka.github.io/2016/09/14/batch_normalization/
        in_coords_key, glob_coords_key = ctx.in_coords_key, ctx.glob_coords_key
        coords_manager = ctx.coords_manager

        # To prevent the memory leakage, compute the norm again
        inv_std, norm_feat = ctx.saved_tensors

        gpool_forward = get_minkowski_function('GlobalPoolingForward', out_grad)
        broadcast_forward = get_minkowski_function('BroadcastForward', out_grad)
        add = operation_type_to_int(OperationType.ADDITION)
        multiply = operation_type_to_int(OperationType.MULTIPLICATION)

        cpp_in_coords_key = in_coords_key.CPPCoordsKey
        cpp_glob_coords_key = glob_coords_key.CPPCoordsKey
        cpp_coords_manager = coords_manager.CPPCoordsManager

        # 1/N \sum dout
        mean_dout, num_nonzero = gpool_forward(out_grad, cpp_in_coords_key,
                                               cpp_glob_coords_key,
                                               cpp_coords_manager, True,
                                               ctx.mode.value)

        # 1/N \sum (dout * out)
        mean_dout_feat, num_nonzero = gpool_forward(out_grad * norm_feat,
                                                    cpp_in_coords_key,
                                                    cpp_glob_coords_key,
                                                    cpp_coords_manager, True,
                                                    ctx.mode.value)

        # out * 1/N \sum (dout * out)
        feat_mean_dout_feat = broadcast_forward(norm_feat, mean_dout_feat,
                                                multiply, cpp_in_coords_key,
                                                cpp_glob_coords_key,
                                                cpp_coords_manager)

        unnorm_din = broadcast_forward(out_grad - feat_mean_dout_feat,
                                       -mean_dout, add, cpp_in_coords_key,
                                       cpp_glob_coords_key, cpp_coords_manager)

        norm_din = broadcast_forward(unnorm_din, inv_std, multiply,
                                     cpp_in_coords_key, cpp_glob_coords_key,
                                     cpp_coords_manager)

        return norm_din, None, None, None, None


class MinkowskiStableInstanceNorm(Module):

    def __init__(self, num_features):
        Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

        self.mean_in = MinkowskiGlobalPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalPooling()
        self.glob_times = MinkowskiBroadcastMultiplication()
        self.reset_parameters()

    def __repr__(self):
        s = f'(nchannels={self.num_features})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        neg_mean_in = self.mean_in(
            SparseTensor(
                -x.F, coords_key=x.coords_key, coords_manager=x.coords_man))
        centered_in = self.glob_sum(x, neg_mean_in)
        temp = SparseTensor(
            centered_in.F**2,
            coords_key=centered_in.coords_key,
            coords_manager=centered_in.coords_man)
        var_in = self.glob_mean(temp)
        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coords_key=var_in.coords_key,
            coords_manager=var_in.coords_man)

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(
            x.F * self.weight + self.bias,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)


class MinkowskiInstanceNorm(Module):
    r"""A instance normalization layer for a sparse tensor.

    """

    def __init__(self, num_features, mode=GlobalPoolingMode.AUTO):
        r"""
        Args:

            num_features (int): the dimension of the input feautres.

            mode (GlobalPoolingModel, optional): The internal global pooling computation mode.
        """
        Module.__init__(self)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.reset_parameters()
        self.mode = mode
        self.inst_norm = MinkowskiInstanceNormFunction()

    def __repr__(self):
        s = f'(nchannels={self.num_features})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, input):
        assert isinstance(input, SparseTensor)

        output = self.inst_norm.apply(input.F, self.mode, input.coords_key,
                                      None, input.coords_man)
        output = output * self.weight + self.bias

        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)
