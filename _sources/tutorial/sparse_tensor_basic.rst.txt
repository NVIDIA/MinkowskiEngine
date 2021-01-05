Sparse Tensor Basics
====================

A sparse tensor is a high-dimensional extension of a sparse matrix where non-zero elements are represented as a set of indices and associated values. Please refer to the `terminology page <https://nvidia.github.io/MinkowskiEngine/terminology.html>`_ for more details.


Data Generation
---------------

One can generate data directly by extracting non-zero elements. Here, we present a simple 2D array with 5 non-zero elements at the center.

.. code-block:: python

   data = [
       [0, 0, 2.1, 0, 0],
       [0, 1, 1.4, 3, 0],
       [0, 0, 4.0, 0, 0]
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

   to_sparse_coo(data)


Note that we extract coordinates along with features. This is a simple example and quite inefficient and artificial. In many real applications, it is unlikely that you will get discretized coordinates. For quantizing and extracting discrete values efficiently, please refer to the `training demo page <https://nvidia.github.io/MinkowskiEngine/demo/training.html>`_.


Sparse Tensor Initialization
----------------------------

The next step in the pipeline is initializing a sparse tensor. A :attr:`MinkowskiEngine.SparseTensor` requires coordinates with batch indices; this results in a sparse tensor with :math:`D+1` spatial dimensions if the original coordinates have :math:`D` dimensions.

.. code-block:: python

   coords0, feats0 = to_sparse_coo(data_batch_0)
   coords1, feats1 = to_sparse_coo(data_batch_1)
   coords, feats = ME.utils.sparse_collate(
       coordinates=[coords0, coords1], features=[feats0, feats1])


Here, we used :attr:`MinkowskiEngine.utils.sparse_collate` function, but you can use :attr:`MinkowskiEngine.utils.batched_coordinates` to convert a list of coordinates to :attr:`MinkowskiEngine.SparseTensor` compatible coordinates.


Sparse Tensor for Continuous Coordinates
----------------------------------------

In many cases, coordinates used in neural networks are continuous.
However, sparse tensors used in sparse tensor networks are defined in a discrete coordinate system.
To convert the features in continuous coordinates to discrete coordinates, we provide feature averaging functions that convert features in continuous coordinates to discrete coordinates.
You can simply use the sparse tensor initialization for this. For example,

.. code-block:: python

   sinput = ME.SparseTensor(
       features=torch.from_numpy(colors), # Convert to a tensor
       coordinates=ME.utils.batched_coordinates([coordinates / voxel_size]),  # coordinates must be defined in a integer grid. If the scale
       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE  # when used with continuous coordinates, average features in the same coordinate
   )
   logits = model(sinput).slice(sinput).F


Please refer to `indoor semantic segmentation <https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/indoor.py>`_ for more detail.


Sparse Tensor Arithmetics
-------------------------

You can use the initialized sparse tensor with a simple feed-forward neural network, but in many cases, you need to do some unconventional operations, and that is why you came to use this library :) Here, we provide some simple operations that allow binary operations between sparse tensors and concatenation along the feature dimension.

.. code-block:: python

   # sparse tensors
   A = ME.SparseTensor(coordinates=coords, features=feats)
   B = ME.SparseTensor(
       coordinates=new_coords,
       features=new_feats,
       coordinate_manager=A.coordinate_manager,  # must share the same coordinate manager
   )

   C = A + B
   C = A - B
   C = A * B
   C = A / B


Here, we create two sparse tensors with different sparsity patterns. However, we forced the second sparse tensor `B` to share the `coordinate_manager`, a coordinate manager. This allows sharing the computation graph between two sparse tensors. The semantics is rather ugly for now, but will be hidden in the future.

If you add two sparse tensors, this will add two features. In case where there is a non-zero element, but not on the other sparse tensor at a specific coordinate, we assume `0` for the non-existing value since a sparse tensor saves non-zero elements only. Anything that we do not specify is `0` by definition. Same goes for all other binary operations.

However, for in-place operations, we force the coordinates to have the same sparsity pattern.

.. code-block:: python

   # in place operations
   # Note that it requires the same coordinate_map_key (no need to feed coords)
   D = ME.SparseTensor(
       # coordinates=coords,  not required
       features=feats,
       coordinate_manager=A.coordinate_manager,  # must share the same coordinate manager
       coordinate_map_key=A.coordinate_map_key  # For inplace, must share the same coords key
   )

   A += D
   A -= D
   A *= D
   A /= D

Note that we use the same `coordinate_map_key` for the sparse tensor `D`. It will give you an assertion error if you try to use a sparse tensor with different `coordinate_map_key`.


Feature Concatenation
---------------------

You can concatenate two sparse tensors along the feature dimension if they share the same `coordinate_map_key`.

.. code-block:: python

   # If you have two or more sparse tensors with the same coordinate_map_key, you can concatenate features
   E = ME.cat(A, D)


Batch-wise Decomposition
------------------------

The internal structure of a sparse tensor collapses all non-zero elements within a batch into a coordinate matrix and a feature matrix.
To decompose the outputs, you can use a couple function and attributes.

.. code-block:: python

   coords0, feats0 = to_sparse_coo(data_batch_0)
   coords1, feats1 = to_sparse_coo(data_batch_1)
   coords, feats = ME.utils.sparse_collate(
       coordinates=[coords0, coords1], features=[feats0, feats1])

   # sparse tensors
   A = ME.SparseTensor(coordinates=coords, features=feats)
   conv = ME.MinkowskiConvolution(
       in_channels=1, out_channels=2, kernel_size=3, stride=2, dimension=2)
   B = conv(A)

   # Extract features and coordinates per batch index
   coords = B.decomposed_coordinates
   feats = B.decomposed_features
   coords, feats = B.decomposed_coordinates_and_features

   # To specify a batch index
   batch_index = 1
   coords = B.coordinates_at(batch_index)
   feats = B.features_at(batch_index)


For more information, please refer to `examples/sparse_tensor_basic.py <https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/sparse_tensor_basic.py>`_.
