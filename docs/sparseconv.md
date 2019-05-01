Generalized Sparse Convolution
==============================

3D data is an readily-available source of input for many robotics, VR/AR/MR applications, as well as autonomous driving. The sensors such as depth cameras, LIDAR sensors can capture the surroundings in 3D.


Sparse Tensor
-------------

A sparse tensor is an N-dimensional extension of a sparse matrix. A sparse matrix is represented as a list of column indices and row indices for non-zero element locations and scalar values associated to each location. If we extend this to an N-dimension, we could uee N-coordinates to represent a location in the space. Also, if we use a vector instead of a scaler for each location, we have N+1 dimensional sparse tensor.


Generalizing the convolution
----------------------------

[Generalized Sparse Convolution](https://arxiv.org/abs/1904.08755)
The convolution is a fundamental operation in many fields. Especially in 2D, the 2D convolution has achieved state-of-the-art performance in many perception tasks. A convolution assumes stationary of the input signal and process 
