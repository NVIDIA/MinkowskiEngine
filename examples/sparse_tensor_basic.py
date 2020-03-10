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
import MinkowskiEngine as ME

data_batch_0 = [
    [0, 0, 2.1, 0, 0],  #
    [0, 1, 1.4, 3, 0],  #
    [0, 0, 4.0, 0, 0]
]

data_batch_1 = [
    [1, 0, 0],  #
    [0, 2, 0],  #
    [0, 0, 3]
]


def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)


def sparse_tensor_initialization():
    coords, feats = to_sparse_coo(data_batch_0)
    # collate sparse tensor data to augment batch indices
    # Note that it is wrapped inside a list!!
    coords, feats = ME.utils.sparse_collate(coords=[coords], feats=[feats])
    sparse_tensor = ME.SparseTensor(coords=coords, feats=feats)


def sparse_tensor_arithmetics():
    coords0, feats0 = to_sparse_coo(data_batch_0)
    coords0, feats0 = ME.utils.sparse_collate(coords=[coords0], feats=[feats0])

    coords1, feats1 = to_sparse_coo(data_batch_1)
    coords1, feats1 = ME.utils.sparse_collate(coords=[coords1], feats=[feats1])

    # sparse tensors
    A = ME.SparseTensor(coords=coords0, feats=feats0)
    B = ME.SparseTensor(coords=coords1, feats=feats1)

    # The following fails
    try:
        C = A + B
    except AssertionError:
        pass

    B = ME.SparseTensor(
        coords=coords1,
        feats=feats1,
        coords_manager=A.coords_man,  # must share the same coordinate manager
        force_creation=True  # must force creation since tensor stride [1] exists
    )

    C = A + B
    C = A - B
    C = A * B
    C = A / B

    # in place operations
    # Note that it requires the same coords_key (no need to feed coords)
    D = ME.SparseTensor(
        # coords=coords,  not required
        feats=feats0,
        coords_manager=A.coords_man,  # must share the same coordinate manager
        coords_key=A.coords_key  # For inplace, must share the same coords key
    )

    A += D
    A -= D
    A *= D
    A /= D

    # If you have two or more sparse tensors with the same coords_key, you can concatenate features
    E = ME.cat(A, D)


def operation_mode():
    # Set to share the coords_man by default
    ME.set_sparse_tensor_operation_mode(
        ME.SparseTensorOperationMode.SHARE_COORDS_MANAGER)
    print(ME.sparse_tensor_operation_mode())

    coords0, feats0 = to_sparse_coo(data_batch_0)
    coords0, feats0 = ME.utils.sparse_collate(coords=[coords0], feats=[feats0])

    coords1, feats1 = to_sparse_coo(data_batch_1)
    coords1, feats1 = ME.utils.sparse_collate(coords=[coords1], feats=[feats1])

    for _ in range(2):
        # sparse tensors
        A = ME.SparseTensor(coords=coords0, feats=feats0)
        B = ME.SparseTensor(
            coords=coords1,
            feats=feats1,
            # coords_manager=A.coords_man,  No need to feed the coords_man
            force_creation=True)

        C = A + B

        # When done using it for forward and backward, you must cleanup the coords man
        ME.clear_global_coords_man()


def decomposition():
    coords0, feats0 = to_sparse_coo(data_batch_0)
    coords1, feats1 = to_sparse_coo(data_batch_1)
    coords, feats = ME.utils.sparse_collate(
        coords=[coords0, coords1], feats=[feats0, feats1])

    # sparse tensors
    A = ME.SparseTensor(coords=coords, feats=feats)
    conv = ME.MinkowskiConvolution(
        in_channels=1, out_channels=2, kernel_size=3, stride=2, dimension=2)
    B = conv(A)

    # Extract features and coordinates per batch index
    list_of_coords = B.decomposed_coordinates
    list_of_feats = B.decomposed_features
    list_of_coords, list_of_feats = B.decomposed_coordinates_and_features

    # To specify a batch index
    batch_index = 1
    coords = B.coordinates_at(batch_index)
    feats = B.features_at(batch_index)

    # Empty list if given an invalid batch index
    batch_index = 3
    print(B.coordinates_at(batch_index))


if __name__ == '__main__':
    sparse_tensor_initialization()
    sparse_tensor_arithmetics()
    operation_mode()
    decomposition()
