Training Pipeline
=================

The Minkowski Engine works seamlessly with the PyTorch
`torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html>`_.
Before you proceed, make sure that you are familiar with the data loading
tutorial `torch.utils.data.DataLoader
<https://pytorch.org/docs/stable/data.html>`_.


Making a Dataset
----------------

The first thing you need to do is loading or generating data. This is the most
time-consuming part if you use your own dataset. However, it is tedious, not
difficult :) The most important part that you need to fill in is
`__getitem__(self, index)` if you inherit `torch.utils.data.Dataset
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_ or
`__iter__(self)` if you inherit `torch.utils.data.IterableDataset
<https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`_.
In this example, let's create a random dataset that generates a noisy line.

.. code-block:: python

   class RandomLineDataset(torch.utils.data.Dataset):

       ...

       def __getitem__(self, i):
           # Regardless of the input index, return randomized data
           angle, intercept = np.tan(self._uniform_to_angle(
               self.rng.rand())), self.rng.rand()

           # Line as x = cos(theta) * t, y = sin(theta) * t + intercept and random t's
           # Drop some samples
           xs_data = self._sample_xs(self.num_data)
           ys_data = angle * xs_data + intercept + self._sample_noise(
               self.num_data, [0, 0.1])

           noise = 4 * (self.rng.rand(self.num_noise, 2) - 0.5)

           # Concatenate data
           input = np.vstack([np.hstack([xs_data, ys_data]), noise])
           feats = input
           labels = np.vstack(
               [np.ones((self.num_data, 1)),
                np.zeros((self.num_noise, 1))]).astype(np.int32)

           ...

           # quantization step
           return various_outputs

Here, I created a dataset that inherits the `torch.utils.data.Dataset
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_, but you
can inherit `torch.utils.data.IterableDataset
<https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`_
and fill out `__iter__(self)` instead.


Quantization
------------

We use a sparse tensor as an input. Like any tensors, a sparse tensor value is
defined at a discrete location (indices). Thus, quantizing the coordinates
whose features are defined is the critical step and `quantization_size` is an
important hyper-parameter that affects the performance of a network
drastically. You must choose the correct quantization size as well as quantize
the coordinate correctly.

The Minkowski Engine provides a set of fast and optimized functions for
quantization and sparse tensor generation. Here, we use
:attr:`MinkowskiEngine.utils.sparse_quantize`.


.. code-block:: python

   class RandomLineDataset(torch.utils.data.Dataset):

       ...

       def __getitem__(self, i): 

           ...

           # Quantize the input
           discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
               coords=input,
               feats=feats,
               labels=labels,
               quantization_size=self.quantization_size)

           return discrete_coords, unique_feats, unique_labels


Another way to quantize a coordinate is to use the returned mapping indices.
This is useful if you have an unconventional input.


.. code-block:: python

   class RandomLineDataset(torch.utils.data.Dataset):

       ...

       def __getitem__(self, i): 

           ...

           coords /= self.quantization_size

           # Quantize the input
           mapping = ME.utils.sparse_quantize(
               coords=coords,
               return_index=True)

           return coords[mapping], feats[mapping], labels[mapping]


Making a DataLoader
-------------------

Once you create your dataset, you need a data loader to call the dataset and
generate a mini-batch for neural network training. This part is relatively
easy, but we have to use a custom `collate_fn
<https://pytorch.org/docs/stable/data.html?highlight=collate_fn#torch.utils.data.DataLoader>`_
to generate a suitable sparse tensor.

.. code-block:: python

   train_dataset = RandomLineDataset(...)
   # Option 1
   train_dataloader = DataLoader(
       train_dataset,
       ...
       collate_fn=ME.utils.SparseCollation())

   # Option 2
   train_dataloader = DataLoader(
       train_dataset,
       ...
       collate_fn=ME.utils.batch_sparse_collate)


Here, we can use the provided collation class
:attr:`MinkowskiEngine.utils.SparseCollation` or the function
:attr:`MinkowskiEngine.utils.batch_sparse_collate` to convert the inputs into
appropriate outputs that we can use to initialize a sparse tensor. However, if
you need your own collation function, you can follow the example below.


.. code-block:: python

   def custom_collation_fn(data_labels):
       coords, feats, labels = list(zip(*data_labels))

       # Create batched coordinates for the SparseTensor input
       bcoords = ME.utils.batched_coordinates(coords)

       # Concatenate all lists
       feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
       labels_batch = torch.from_numpy(np.concatenate(labels, 0)).int()

       return bcoords, feats, labels

   ...

   train_dataset = RandomLineDataset(...)
   train_dataloader = DataLoader(
       train_dataset,
       ...
       collate_fn=custom_collation_fn)


Training
--------

Once you have everything, let's create a network and train it with the
generated data. One thing to note is that if you use more than one
:attr:`num_workers` for the data loader, you have to make sure that the
:attr:`MinkowskiEngine.SparseTensor` generation part has to be located within the main python
process since all python multi-processes use separate processes and the
:attr:`MinkowskiEngine.CoordsManager`, the
internal C++ structure that maintains the coordinate hash tables and kernel
maps, cannot be referenced outside the process that generated it.

.. code-block:: python

   # Binary classification
   net = UNet(
       2,  # in nchannel
       2,  # out_nchannel
       D=2)

   optimizer = optim.SGD(
       net.parameters(),
       lr=config.lr,
       momentum=config.momentum,
       weight_decay=config.weight_decay)

   criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

   # Dataset, data loader
   train_dataset = RandomLineDataset(noise_type='gaussian')

   train_dataloader = DataLoader(
       train_dataset,
       batch_size=config.batch_size,
       collate_fn=collation_fn,
       num_workers=1)

   for epoch in range(config.max_epochs):
       train_iter = iter(train_dataloader)

       # Training
       net.train()
       for i, data in enumerate(train_iter):
           coords, feats, labels = data
           out = net(ME.SparseTensor(feats, coords))
           optimizer.zero_grad()
           loss = criterion(out.F.squeeze(), labels.long())
           loss.backward()
           optimizer.step()

           accum_loss += loss.item()
           accum_iter += 1
           tot_iter += 1

           if tot_iter % 10 == 0 or tot_iter == 1:
               print(
                   f'Iter: {tot_iter}, Epoch: {epoch}, Loss: {accum_loss / accum_iter}'
               )
               accum_loss, accum_iter = 0, 0


Finally, once you assemble all the codes, you can train your network.

::

   $ python -m examples.training

   Epoch: 0 iter: 1, Loss: 0.7992178201675415
   Epoch: 0 iter: 10, Loss: 0.5555745628145006
   Epoch: 0 iter: 20, Loss: 0.4025680094957352
   Epoch: 0 iter: 30, Loss: 0.3157463788986206
   Epoch: 0 iter: 40, Loss: 0.27348957359790804
   Epoch: 0 iter: 50, Loss: 0.2690591633319855
   Epoch: 0 iter: 60, Loss: 0.258208692073822
   Epoch: 0 iter: 70, Loss: 0.34842072874307634
   Epoch: 0 iter: 80, Loss: 0.27565130293369294
   Epoch: 0 iter: 90, Loss: 0.2860450878739357
   Epoch: 0 iter: 100, Loss: 0.24737665355205535
   Epoch: 1 iter: 110, Loss: 0.2428090125322342
   Epoch: 1 iter: 120, Loss: 0.25397603064775465
   Epoch: 1 iter: 130, Loss: 0.23624965399503708
   Epoch: 1 iter: 140, Loss: 0.2247777447104454
   Epoch: 1 iter: 150, Loss: 0.22956613600254058
   Epoch: 1 iter: 160, Loss: 0.22803852707147598
   Epoch: 1 iter: 170, Loss: 0.24081039279699326
   Epoch: 1 iter: 180, Loss: 0.22322929948568343
   Epoch: 1 iter: 190, Loss: 0.22531934976577758
   Epoch: 1 iter: 200, Loss: 0.2116936132311821
   ...

The original code can be found at `examples/training.py
<https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/training.py>`_.
