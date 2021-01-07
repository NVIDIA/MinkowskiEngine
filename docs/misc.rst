Miscellanea
===========

Controlling the number of threads
---------------------------------

The kernel map `[1] <https://arxiv.org/abs/1904.08755>`_ defines which row of an input feature matrix to which row of the output feature matrix. This however is an expensive operation as the dimension increases. Fortunately, some part of the operation can be parallelized and we provide a multi-threaded function to speed up this process.

By default, we use all CPU threads available in the system. However, this might not be desirable in some cases. Simply define an environmental variable ``OMP_NUM_THREADS`` to control the number of threads you want to use. For example, ``export OMP_NUM_THREADS=8; python your_program.py``. If you use SLURM, the environment variable ``OMP_NUM_THREADS`` will be automatically set.



is_cuda_available
-----------------

.. autofunction:: MinkowskiEngine.is_cuda_available


cuda_version
------------

.. autofunction:: MinkowskiEngine.cuda_version


get_gpu_memory_info
-------------------

.. autofunction:: MinkowskiEngine.get_gpu_memory_info


set_memory_manager_backend
--------------------------

.. autofunction:: MinkowskiEngine.set_memory_manager_backend




References
----------

- `[1] 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19 <https://arxiv.org/abs/1904.08755>`_
