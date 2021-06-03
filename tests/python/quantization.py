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
import unittest
import numpy as np

from MinkowskiEngine.utils import sparse_quantize
import MinkowskiEngineBackend._C as MEB


class TestQuantization(unittest.TestCase):
    def test(self):
        N = 16575
        ignore_label = 255

        coords = np.random.rand(N, 3) * 100
        feats = np.random.rand(N, 4)
        labels = np.floor(np.random.rand(N) * 3)

        labels = labels.astype(np.int32)

        # Make duplicates
        coords[:3] = 0
        labels[:3] = 2
        quantized_coords, quantized_feats, quantized_labels = sparse_quantize(
            coords.astype(np.int32), feats, labels, ignore_label
        )

        print(quantized_labels)

    def test_device(self):
        N = 16575
        coords = np.random.rand(N, 3) * 100

        # Make duplicates
        coords[:3] = 0
        unique_map = sparse_quantize(
            coords.astype(np.int32), return_maps_only=True, device="cpu"
        )

        print(len(unique_map))
        unique_map = sparse_quantize(
            coords.astype(np.int32), return_maps_only=True, device="cuda"
        )
        print(len(unique_map))

    def test_mapping(self):
        N = 16575
        coords = (np.random.rand(N, 3) * 100).astype(np.int32)
        mapping, inverse_mapping = MEB.quantize_np(coords)
        print("N unique:", len(mapping), "N:", N)
        self.assertTrue((coords == coords[mapping][inverse_mapping]).all())
        self.assertTrue((coords == coords[mapping[inverse_mapping]]).all())

        coords = torch.from_numpy(coords)
        mapping, inverse_mapping = MEB.quantize_th(coords)
        print("N unique:", len(mapping), "N:", N)
        self.assertTrue((coords == coords[mapping[inverse_mapping]]).all())

        unique_coords, index, reverse_index = sparse_quantize(
            coords, return_index=True, return_inverse=True
        )
        self.assertTrue((coords == coords[index[reverse_index]]).all())

    def test_label_np(self):
        N = 16575

        coords = (np.random.rand(N, 3) * 100).astype(np.int32)
        labels = np.floor(np.random.rand(N) * 3).astype(np.int32)

        # Make duplicates
        coords[:3] = 0
        labels[:3] = 2

        mapping, inverse_mapping, colabel = MEB.quantize_label_np(coords, labels, -1)
        self.assertTrue(np.sum(np.abs(coords[mapping][inverse_mapping] - coords)) == 0)
        self.assertTrue(np.sum(colabel < 0) > 3)

    def test_collision(self):
        coords = np.array([[0, 0], [0, 0], [0, 0], [0, 1]], dtype=np.int32)
        labels = np.array([0, 1, 2, 3], dtype=np.int32)

        unique_coords, colabels = sparse_quantize(
            coords, labels=labels, ignore_label=255
        )
        print(unique_coords)
        print(colabels)
        self.assertTrue(len(unique_coords) == 2)
        self.assertTrue(np.array([0, 0]) in unique_coords)
        self.assertTrue(np.array([0, 1]) in unique_coords)
        self.assertTrue(len(colabels) == 2)
        self.assertTrue(255 in colabels)

    def test_quantization_size(self):
        coords = torch.randn((1000, 3), dtype=torch.float)
        feats = torch.randn((1000, 10), dtype=torch.float)
        res = sparse_quantize(coords, feats, quantization_size=0.1)
        print(res[0].shape, res[1].shape)
        res = sparse_quantize(coords.numpy(), feats.numpy(), quantization_size=0.1)
        print(res[0].shape, res[1].shape)


if __name__ == "__main__":
    unittest.main()
