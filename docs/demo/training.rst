Training Pipeline with Pytorch DataLoader
=========================================

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
               xs_data, [0, 1])

           # Concatenate data
           input = np.hstack([xs_data, ys_data])
           feats = np.random.rand(self.num_data, 1)
           labels = np.ones((self.num_data, 1))

           # Discretize
           discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
               coords=input,
               feats=feats,
               labels=labels,
               hash_type='ravel',
               set_ignore_label_when_collision=False,
               quantization_size=self.quantization_size)
           return discrete_coords, unique_feats, unique_labels


Here, I created a dataset that inherits the `torch.utils.data.Dataset
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_, but you
can inherit `torch.utils.data.IterableDataset
<https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`_
and fill out `__iter__(self)` instead.


Making a DataLoader
-------------------

Once you create your dataset, you need a data loader to call the dataset and
generate a mini-batch for neural network training. This part is relatively
easy, but we have to use a custom `collate_fn
<https://pytorch.org/docs/stable/data.html?highlight=collate_fn#torch.utils.data.DataLoader>`_
to generate a suitable sparse tensor.

.. code-block:: python

   train_dataset = RandomLineDataset(...)
   train_dataloader = torch.utils.data.DataLoader(
       train_dataset, batch_size=config.batch_size, collate_fn=collation_fn)

Here, we used our custom :attr:`collation_fn`. The collation function has to
concatenate all sparse tensors generate from each call that generates a batch
and assign the correct batch index to the coordinates.

.. code-block:: python

   def collation_fn(data_labels):
       coords, feats, labels = list(zip(*data_labels))
       coords_batch, feats_batch, labels_batch = [], [], []

       for batch_id, _ in enumerate(coords):
           N = coords[batch_id].shape[0]

           coords_batch.append(
               torch.cat((torch.from_numpy(
                   coords[batch_id]).int(), torch.ones(N, 1).int() * batch_id), 1))
           feats_batch.append(torch.from_numpy(feats[batch_id]))
           labels_batch.append(torch.from_numpy(labels[batch_id]))

       # Concatenate all lists
       coords_batch = torch.cat(coords_batch, 0).int()
       feats_batch = torch.cat(feats_batch, 0).float()
       labels_batch = torch.cat(labels_batch, 0).float()

       return coords_batch, feats_batch, labels_batch

Training
--------

Once you have everything, let's create a network and train it with the
generated data. One thing to note is that if you use more than one
:attr:`num_workers` for the data loader, you have to make sure that the
:attr:`ME.SparseTensor` generation part has to be located within the main python
process since all python multi-processes use separate processes and the
`ME.CoordManager
<https://stanfordvl.github.io/MinkowskiEngine/coords.html#coordsmanager>`_, the
internal C++ structure that maintains the coordinate hash tables and kernel
maps, cannot be referenced outside the process that generated it.

.. code-block:: python

   net = UNet(1, 1, D=2)
   optimizer = optim.SGD(
       net.parameters(),
       lr=config.lr,
       momentum=config.momentum,
       weight_decay=config.weight_decay)
   binary_crossentropy = torch.nn.BCEWithLogitsLoss()
   accum_loss, accum_iter, tot_iter = 0, 0, 0

   for epoch in range(config.max_epochs):
       train_iter = train_dataloader.__iter__()

       # Training
       net.train()
       for i, data in enumerate(train_iter):
           coords, feats, labels = data
           out = net(ME.SparseTensor(feats, coords))
           optimizer.zero_grad()
           loss = binary_crossentropy(out.F, labels)
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

   $ python -m examples.two_dim_training
   Iter: 1, Epoch: 0, Loss: 0.8510904908180237
   Iter: 10, Epoch: 2, Loss: 0.4347594661845101
   Iter: 20, Epoch: 4, Loss: 0.02069884107913822
   Iter: 30, Epoch: 7, Loss: 0.0010139490244910122
   Iter: 40, Epoch: 9, Loss: 0.0003139576627290808
   Iter: 50, Epoch: 12, Loss: 0.000194330868544057
   Iter: 60, Epoch: 14, Loss: 0.00015514824335696175
   Iter: 70, Epoch: 17, Loss: 0.00014614587998948992
   Iter: 80, Epoch: 19, Loss: 0.00013127068668836728


The original code can be found at `example/two_dim_training.py
<https://github.com/StanfordVL/MinkowskiEngine/blob/master/examples/two_dim_training.py>`_.
