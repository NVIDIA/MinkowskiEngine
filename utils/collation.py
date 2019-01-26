import torch
import logging


def sparse_collate(coords, feats, labels=None):
    """Create torch matrices for coordinates with batch indices and features
    """
    use_label = False if labels is None else True
    coords_batch, feats_batch, labels_batch = [], [], []

    batch_id = 0
    for batch_id, _ in enumerate(coords):
        num_points = coords[batch_id].shape[0]
        coords_batch.append(
            torch.cat((torch.from_numpy(coords[batch_id]).int(),
                       torch.ones(num_points, 1).int() * batch_id), 1))
        feats_batch.append(torch.from_numpy(feats[batch_id]))
        if use_label:
            labels_batch.append(torch.from_numpy(labels[batch_id]))
        batch_id += 1

    # Concatenate all lists
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats_batch, 0).float()
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
