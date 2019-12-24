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
import numpy as np
import torch
import logging
import collections.abc


def batched_coordinates(coords):
    r"""Create a `ME.SparseTensor` coordinates from a sequence of coordinates

    Given a list of either numpy or pytorch tensor coordinates, return the
    batched coordinates suitable for `ME.SparseTensor`.

    Args:
        :attr:`coords` (a sequence of `torch.Tensor` or `numpy.ndarray`): a
        list of coordinates.

    Returns:
        :attr:`coords` (`torch.IntTensor`): a batched coordinates.

    .. warning::

       From v0.4, the batch index will be prepended before all coordinates.

    """
    assert isinstance(
        coords, collections.abc.Sequence), "The coordinates must be a sequence."
    assert np.array([cs.ndim == 2 for cs in coords]).all(), \
        "All coordinates must be in a 2D array."
    D = np.unique(np.array([cs.shape[1] for cs in coords]))
    assert len(D) == 1, f"Dimension of the array mismatch. All dimensions: {D}"
    D = D[0]

    # Create a batched coordinates
    N = np.array([len(cs) for cs in coords]).sum()
    bcoords = torch.IntTensor(N, D + 1)  # uninitialized

    # if not BATCH_FIRST:
    s = 0
    for b, cs in enumerate(coords):
        if isinstance(cs, np.ndarray):
            cs = torch.from_numpy(np.floor(cs).astype(np.int32))
        else:
            cs = cs.floor().int()
        cn = len(cs)
        bcoords[s:s + cn, :D] = cs
        bcoords[s:s + cn, D] = b
        s += cn
    return bcoords


def sparse_collate(coords, feats, labels=None):
    r"""Create input arguments for a sparse tensor `the documentation
    <https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html>`_.

    Convert a set of coordinates and features into the batch coordinates and
    batch features.

    Args:
        :attr:`coords` (set of `torch.Tensor` or `numpy.ndarray`): a set of coordinates.

        :attr:`feats` (set of `torch.Tensor` or `numpy.ndarray`): a set of features.

        :attr:`labels` (set of `torch.Tensor` or `numpy.ndarray`): a set of labels
        associated to the inputs.

    """
    use_label = False if labels is None else True
    feats_batch, labels_batch = [], []
    assert isinstance(coords, collections.abc.Sequence), \
            "The coordinates must be a sequence of arrays or tensors."
    assert isinstance(feats, collections.abc.Sequence), \
            "The features must be a sequence of arrays or tensors."
    D = np.unique(np.array([cs.shape[1] for cs in coords]))
    assert len(D) == 1, f"Dimension of the array mismatch. All dimensions: {D}"
    D = D[0]

    if use_label:
        assert isinstance(labels, collections.abc.Sequence), \
            "The labels must be a sequence of arrays or tensors."

    N = np.array([len(cs) for cs in coords]).sum()
    Nf = np.array([len(fs) for fs in feats]).sum()
    assert N == Nf, f"Coordinate length {N} != Feature length {Nf}"

    batch_id = 0
    s = 0  # start index
    bcoords = torch.IntTensor(N, D + 1)  # uninitialized batched coords
    for coord, feat in zip(coords, feats):
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord)
        else:
            assert isinstance( coord, torch.Tensor), \
                "Coords must be of type numpy.ndarray or torch.Tensor"
        coord = coord.int()

        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        else:
            assert isinstance( feat, torch.Tensor), \
                "Features must be of type numpy.ndarray or torch.Tensor"

        # Labels
        if use_label:
            label = labels[batch_id]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            else:
                assert isinstance(label, torch.Tensor), \
                    "labels must be of type numpy.ndarray or torch.Tensor"
            labels_batch.append(label)

        # Batched coords
        cn = coord.shape[0]
        bcoords[s:s + cn, :D] = coord
        bcoords[s:s + cn, D] = batch_id

        # Features
        feats_batch.append(feat)

        # Post processing steps
        batch_id += 1
        s += cn

    # Concatenate all lists
    feats_batch = torch.cat(feats_batch, 0)
    if use_label:
        labels_batch = torch.cat(labels_batch, 0)
        return bcoords, feats_batch, labels_batch
    else:
        return bcoords, feats_batch


def batch_sparse_collate(data):
    r"""The wrapper function that can be used in in conjunction with
    `torch.utils.data.DataLoader` to generate inputs for a sparse tensor.

    Please refer to `the training example
    <https://stanfordvl.github.io/MinkowskiEngine/demo/training.html>`_ for the
    usage.

    Args:
        :attr:`data`: list of (coordinates, features, labels) tuples.

    """
    return sparse_collate(*list(zip(*data)))


class SparseCollation:
    r"""Generates collate function for coords, feats, labels.

    Please refer to `the training example
    <https://stanfordvl.github.io/MinkowskiEngine/demo/training.html>`_ for the
    usage.

    Args:
        :attr:`limit_numpoints` (int): If positive integer, limits batch size
        so that the number of input coordinates is below limit_numpoints. If 0
        or False, concatenate all points. -1 by default.

    Example::

        >>> data_loader = torch.utils.data.DataLoader(
        >>>     dataset,
        >>>     ...,
        >>>     collate_fn=SparseCollation())
        >>> for d in iter(data_loader):
        >>>     print(d)

    """

    def __init__(self, limit_numpoints=-1):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch = [], [], []

        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints > 0 and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\tCannot fit {num_full_points} points into'
                    ' {self.limit_numpoints} points limit. Truncating batch '
                    f'size at {batch_id} out of {num_full_batch_size} with '
                    f'{batch_num_points - num_points}.')
                break
            coords_batch.append(coords[batch_id])
            feats_batch.append(feats[batch_id])
            labels_batch.append(labels[batch_id])

        # Concatenate all lists
        return sparse_collate(coords_batch, feats_batch, labels_batch)
