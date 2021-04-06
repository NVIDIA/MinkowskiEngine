# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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

import os
import numpy as np
import collections
from urllib.request import urlretrieve

import torch

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

if not os.path.isfile("1.ply"):
    urlretrieve("https://bit.ly/3c2iLhg", "1.ply")


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


def batched_coordinates(coords, dtype=torch.int32, device=None):
    r"""Create a `ME.SparseTensor` coordinates from a sequence of coordinates

    Given a list of either numpy or pytorch tensor coordinates, return the
    batched coordinates suitable for `ME.SparseTensor`.

    Args:
        :attr:`coords` (a sequence of `torch.Tensor` or `numpy.ndarray`): a
        list of coordinates.

        :attr:`dtype`: torch data type of the return tensor. torch.int32 by default.

    Returns:
        :attr:`batched_coordindates` (`torch.Tensor`): a batched coordinates.

    .. warning::

       From v0.4, the batch index will be prepended before all coordinates.

    """
    assert isinstance(
        coords, collections.abc.Sequence
    ), "The coordinates must be a sequence."
    assert np.array(
        [cs.ndim == 2 for cs in coords]
    ).all(), "All coordinates must be in a 2D array."
    D = np.unique(np.array([cs.shape[1] for cs in coords]))
    assert len(D) == 1, f"Dimension of the array mismatch. All dimensions: {D}"
    D = D[0]
    if device is None:
        if isinstance(coords, torch.Tensor):
            device = coords[0].device
        else:
            device = "cpu"
    assert dtype in [
        torch.int32,
        torch.float32,
    ], "Only torch.int32, torch.float32 supported for coordinates."

    # Create a batched coordinates
    N = np.array([len(cs) for cs in coords]).sum()
    bcoords = torch.zeros((N, D + 1), dtype=dtype, device=device)  # uninitialized

    s = 0
    for b, cs in enumerate(coords):
        if dtype == torch.int32:
            if isinstance(cs, np.ndarray):
                cs = torch.from_numpy(np.floor(cs))
            elif not (
                isinstance(cs, torch.IntTensor) or isinstance(cs, torch.LongTensor)
            ):
                cs = cs.floor()

            cs = cs.int()
        else:
            if isinstance(cs, np.ndarray):
                cs = torch.from_numpy(cs)

        cn = len(cs)
        # BATCH_FIRST:
        bcoords[s : s + cn, 1:] = cs
        bcoords[s : s + cn, 0] = b
        s += cn
    return bcoords