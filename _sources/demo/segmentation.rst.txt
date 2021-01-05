Semantic Segmentation
=====================

To run the example, please install `Open3D <http://www.open3d.org/>`_ with `pip
install open3d-python`.

.. code-block:: shell

   cd /path/to/MinkowskiEngine
   python -m examples.indoor


Segmentation of a hotel room
----------------------------

When you run the example, you will see a hotel room and semantic segmentation
of the room. You can interactively rotate the visualization when you run the
example. First, we load the data.

.. code-block:: python

    def load_file(file_name):
        pcd = o3d.read_point_cloud(file_name)
        coords = np.array(pcd.points)
        colors = np.array(pcd.colors)
        return coords, colors, pcd

You can provide a quantized coordinates that ensures there would be only one point per voxel, or you can use the new :attr:`MinkowskiEngine.TensorField` that does not require quantized coordinates to process point clouds. However, since it does the quanization in the main training process instead of delegating the quantization to the data loading processes, it could slow down the training.
Next, you should create a batched coordinates by calling :attr:`MinkowskiEngine.utils.batched_coordinates`.

.. code-block:: python

   # Create a batch, this process is done in a data loader during training in parallel.
   in_field = ME.TensorField(
       features=torch.from_numpy(colors).float(),
       coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
       minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
       device=device,
   )


Finally, we feed-forward the sparse tensor into the network and get the predictions.


.. code-block:: python

   # Convert to a sparse tensor
   sinput = in_field.sparse()
   # Output sparse tensor
   soutput = model(sinput)
   # get the prediction on the input tensor field
   out_field = soutput.slice(in_field)


After doing some post-processing. We can color the labels and visualize the
input and the prediction side-by-side.

.. image:: ../images/segmentation.png


The weights are downloaded automatically once you run the example and the
weights are currently the top-ranking algorithm on the `Scannet 3D segmentation
benchmark <http://kaldir.vc.in.tum.de/scannet_benchmark/>`_.

Please refer to the `complete indoor segmentation example
<https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/indoor.py>`_
for more detail.
