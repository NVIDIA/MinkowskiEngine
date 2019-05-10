import numpy as np
import torch
import logging


def sparse_collate(coords, feats, labels=None, is_double=False):
    r"""Create a sparse tensor with batch indices C in `the documentation
    <https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html>`_.

    Convert a set of coordinates and features into the batch coordinates and
    batch features.

    Args:
        coords (set of `torch.Tensor` or `numpy.ndarray`): a set of coordinates.

        feats (set of `torch.Tensor` or `numpy.ndarray`): a set of features.

        labels (set of `torch.Tensor` or `numpy.ndarray`): a set of labels
        associated to the inputs.

        is_double (`bool`): return double precision features if True. False by
        default.

    """
    use_label = False if labels is None else True
    coords_batch, feats_batch, labels_batch = [], [], []

    batch_id = 0
    for coord, feat in zip(coords, feats):
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord)
        else:
            assert isinstance(
                coord, torch.Tensor
            ), "Coords must be of type numpy.ndarray or torch.Tensor"
        coord = coord.int()

        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        else:
            assert isinstance(
                feat, torch.Tensor
            ), "Features must be of type numpy.ndarray or torch.Tensor"
        feat = feat.double() if is_double else feat.float()

        # Batched coords
        num_points = coord.shape[0]
        coords_batch.append(
            torch.cat((coord, torch.ones(num_points, 1).int() * batch_id), 1))

        # Features
        feats_batch.append(feat)

        # Labels
        if use_label:
            label = labels[batch_id]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            else:
                assert isinstance(
                    label, torch.Tensor
                ), "labels must be of type numpy.ndarray or torch.Tensor"
            labels_batch.append(label)

        batch_id += 1

    # Concatenate all lists
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats_batch, 0)
    if use_label:
        labels_batch = torch.cat(labels_batch, 0)
        return coords_batch, feats_batch, labels_batch
    else:
        return coords_batch, feats_batch


class SparseCollation:
    """Generates collate function for coords, feats, labels.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive
        integer, limits batch size so that the number of input
        coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch = [], [], []

        batch_id = 0
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
            coords_batch.append(
                torch.cat((torch.from_numpy(coords[batch_id]).int(),
                           torch.ones(num_points, 1).int() * batch_id), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]))

            batch_id += 1

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0)  # arbitrary format
        return coords_batch, feats_batch, labels_batch
