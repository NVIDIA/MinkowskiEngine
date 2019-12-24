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
import MinkowskiEngineBackend as MEB


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
            coords.astype(np.int32), feats, labels, ignore_label)

        print(quantized_labels)

    def test_mapping(self):
        N = 16575
        coords = (np.random.rand(N, 3) * 100).astype(np.int32)
        mapping = MEB.quantize_np(coords)
        print('N unique:', len(mapping), 'N:', N)

        mapping = MEB.quantize_th(torch.from_numpy(coords))
        print('N unique:', len(mapping), 'N:', N)

    def test_label(self):
        N = 16575
        ignore_label = 255

        coords = (np.random.rand(N, 3) * 100).astype(np.int32)
        feats = np.random.rand(N, 4)
        labels = np.floor(np.random.rand(N) * 3)

        labels = labels.astype(np.int32)

        # Make duplicates
        coords[:3] = 0
        labels[:3] = 2

        mapping, colabels = MEB.quantize_label_np(coords, labels, ignore_label)
        print('Unique labels and counts:',
              np.unique(colabels, return_counts=True))
        print('N unique:', len(mapping), 'N:', N)

        mapping, colabels = MEB.quantize_label_th(
            torch.from_numpy(coords), torch.from_numpy(labels), ignore_label)
        print('Unique labels and counts:',
              np.unique(colabels, return_counts=True))
        print('N unique:', len(mapping), 'N:', N)

        qcoords, qfeats, qlabels = sparse_quantize(coords, feats, labels,
                                                   ignore_label)
        self.assertTrue(len(mapping) == len(qcoords))

    def test_collision(self):
        coords = np.array([[0, 0], [0, 0], [0, 0], [0, 1]], dtype=np.int32)
        labels = np.array([0, 1, 2, 3], dtype=np.int32)

        mapping, colabels = sparse_quantize(
            coords, labels=labels, ignore_label=255)
        print(mapping)
        print(colabels)

        coords = np.array([[0, 0], [0, 1]], dtype=np.int32)
        discrete_coords = sparse_quantize(coords)
        print(discrete_coords)
        discrete_coords = sparse_quantize(torch.from_numpy(coords))
        print(discrete_coords)

    def test_quantization_size(self):
        coords = torch.randn((1000, 3), dtype=torch.float)
        feats = torch.randn((1000, 10), dtype=torch.float)
        res = sparse_quantize(coords, feats, quantization_size=0.1)
        print(res[0].shape, res[1].shape)
        res = sparse_quantize(coords.numpy(), feats.numpy(), quantization_size=0.1)
        print(res[0].shape, res[1].shape)


if __name__ == '__main__':
    unittest.main()
