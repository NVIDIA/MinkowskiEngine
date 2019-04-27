import torch
import itertools
import numpy as np
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF


class Octree:

    def __init__(self, coords, max_depth=5):
        """
        Args:
          coords: 3D pointcloud coordinates. numpy array of shape (num_points, 3)
          max_depth: maximum depth from leaf to root. positive integer.
        """
        # Check input.
        if max_depth <= 0:
            raise ValueError('max_depth should be a positive integer.')
        num_coords = coords.shape[0]
        if coords.shape[1] != 3:
            raise ValueError('Input shape mismatch')

        self.level_map = np.zeros((max_depth, num_coords), dtype=np.int32)
        self.level_coords = []
        self.level_empty_coords = []
        coords_base = coords.min(0) - 2**max_depth
        coords_base -= coords_base % (2**max_depth)
        coords_based = coords - coords_base
        for d in range(max_depth):
            level_size = 2**(max_depth - d)
            level_dim = coords_based.max(0) // level_size + 1
            level_idx = np.ravel_multi_index(coords_based.T // level_size,
                                             level_dim)
            unique_idx, inverse_idx = np.unique(level_idx, return_inverse=True)
            self.level_map[d] = inverse_idx
            curr_level_coords = np.array(
                np.unravel_index(unique_idx, level_dim)).T
            curr_level_coords = (
                curr_level_coords * level_size + coords_base).astype(np.int32)
            self.level_coords.append(curr_level_coords)
        self.level_coords.append(coords)
        coords_dim = coords_based.max(0) + 2**max_depth + 1
        for d in range(max_depth):
            level_size = 2**(max_depth - d - 2)
            iteration_idx = np.array(
                list(itertools.product((0, int(level_size * 2)), repeat=3)))
            iteration_idx = np.ravel_multi_index(iteration_idx.T, coords_dim)
            curr_idx = np.ravel_multi_index(
                (self.level_coords[d] - coords_base).T, coords_dim)
            next_idx = set(
                np.ravel_multi_index((self.level_coords[d + 1] - coords_base).T,
                                     coords_dim))
            curr_expand_idx = set((np.tile(curr_idx,
                                           (8, 1)).T + iteration_idx).flatten())
            assert curr_expand_idx.intersection(next_idx) == next_idx
            next_empty_idx = np.array(list(curr_expand_idx - next_idx))
            next_empty_coords = np.array(
                np.unravel_index(next_empty_idx, coords_dim)).T + coords_base
            self.level_empty_coords.append(next_empty_coords)


class GUNet(ME.MinkowskiNetwork):

    def __init__(self, in_nchannel, out_nchannel, D):
        super(GUNet, self).__init__(D)
        # new forced coords at tensor_stride 2
        self.gcoords_key_s2 = ME.CoordsKey(D)
        self.gcoords_key_s2.setTensorStride(2)
        # new forced coords at tensor stride 1
        self.gcoords_key_s1 = ME.CoordsKey(D)
        self.gcoords_key_s1.setTensorStride(1)

        self.gcoords_s2 = None  # New forced coordinates for tensor stride 2
        self.gcoords_s1 = None  # New forced coordinates for tensor stride 1

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_nchannel,
            out_channels=8,
            kernel_size=3,
            stride=1,
            dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(8)
        self.unpool1 = ME.MinkowskiPoolingTranspose(
            kernel_size=2,
            stride=1,
            out_coords_key=self.gcoords_key_s1,
            dimension=D)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(16)
        self.unpool2 = ME.MinkowskiPoolingTranspose(
            kernel_size=2,
            stride=1,
            out_coords_key=self.gcoords_key_s2,
            dimension=D)
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(32)
        self.conv4 = ME.MinkowskiConvolutionTranspose(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2,
            out_coords_key=self.gcoords_key_s2,
            dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(16)
        self.conv5 = ME.MinkowskiConvolutionTranspose(
            in_channels=32,
            out_channels=8,
            kernel_size=2,
            stride=2,
            out_coords_key=self.gcoords_key_s1,
            dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(8)

        self.conv6 = ME.MinkowskiConvolution(
            in_channels=16,
            out_channels=out_nchannel,
            kernel_size=1,
            stride=1,
            dimension=D)

    def initialize(self, gcoords_s2, gcoords_s1):
        self.gcoords_s2 = gcoords_s2
        self.gcoords_s1 = gcoords_s1

    def forward(self, x):
        assert self.gcoords_s2 is not None
        assert self.gcoords_s1 is not None

        out_s1 = self.bn1(self.conv1(x))
        out = MF.relu(out_s1)

        out_s2 = self.bn2(self.conv2(out))
        out = MF.relu(out_s2)

        out_s4 = self.bn3(self.conv3(out))
        out = MF.relu(out_s4)

        # Create new coords
        x.coords_man.initialize_enforce(self.gcoords_s2, self.gcoords_key_s2)
        x.coords_man.initialize_enforce(self.gcoords_s1, self.gcoords_key_s1)

        out_s1 = self.unpool1(out_s1)
        out_s2 = self.unpool2(out_s2)

        out = MF.relu(self.bn4(self.conv4(out)))
        out = ME.cat((out, out_s2))

        out = MF.relu(self.bn5(self.conv5(out)))
        out = ME.cat((out, out_s1))

        return self.conv6(out)


if __name__ == '__main__':
    in_nchannel = 3

    input_coords = np.array(((0, 1, 0), (3, 0, 0)))
    input_feats = np.random.randn(input_coords.shape[0], in_nchannel)
    output_coords = np.array(((0, 1, 0), (2, 1, 0), (0, 0, 0), (1, 1, 0)))
    octree = Octree(output_coords, max_depth=1)

    input_s2 = octree.level_coords[0]
    input_s1 = octree.level_coords[1]

    print(f'input_s2: {input_s2}')
    print(f'input_s1: {input_s1}')

    input_s2 = np.hstack((input_s2, np.zeros((input_s2.shape[0], 1))))
    input_s1 = np.hstack((input_s1, np.zeros((input_s1.shape[0], 1))))
    input_coords = np.hstack((input_coords, np.zeros((input_coords.shape[0],
                                                      1))))
    output_coords = np.hstack((output_coords,
                               np.zeros((output_coords.shape[0], 1))))

    input_s2 = torch.from_numpy(input_s2).int()
    input_s1 = torch.from_numpy(input_s1).int()
    input_coords = torch.from_numpy(input_coords).int()
    output_coords = torch.from_numpy(output_coords).int()
    input_feats = torch.from_numpy(input_feats).float()

    net = GUNet(in_nchannel, 5, D=3)
    net.eval()
    sinput = ME.SparseTensor(input_feats, input_coords)
    net.initialize(input_s2, input_s1)
    out = net(sinput)
    print(out)
