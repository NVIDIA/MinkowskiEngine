Definitions and Terminology
===========================

Sparse Tensor
-------------

A sparse tensor is a high-dimensional extension of a sparse matrix where non-zero elements are represented as a set of indices and associated values. We use the COOrdinate list (COO) format to save a sparse tensor `[1] <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_. This representation is simply a concatenation of coordinates into a matrix :math:`C` and associated values or features :math:`F`.

In Minkowski Engine, we allow negative indices (or coordinates) and associated values are vectors.

.. math::

   \mathbf{C} = \begin{bmatrix}
   x_1^1   & x_1^2  & \cdots & x_1^D  \\
    \vdots & \vdots & \ddots & \vdots \\
   x_N^1   & x_N^2  & \cdots & x_N^D
   \end{bmatrix}, \; \mathbf{F} = \begin{bmatrix}
   \mathbf{f}_1^T\\
   \vdots\\
   \mathbf{f}_N^T
   \end{bmatrix}

The indices :math:`C` are the column-wise concatenation of indices of non-zero values. In sum, a sparse tensor consists of a set of coordinates :math:`C \in \mathbb{Z}^{N \times D}` and associated features :math:`F \in \mathbb{R}^{N \times N_F}` where :math:`N` is the number of non-zero elements within a sparse tensor, :math:`D` is the dimension of the space, and :math:`N_F` is the number of channels.


Tensor Stride
-------------

In 2D ConvNets, convolution/pooling layers have stride size. This allows the neural networks to have exponentially large receptive field size as well as shrink down the space the convolution/pooling layers operate on. For example, if we apply a stride 2 convolution on a 32 :math:`\times` 32 pixel image, we get 16 :math:`\times` 16 resolution image as an output. The pixels in 16 :math:`\times` 16 resolution image have stride size 2 between neighboring pixels. Likewise, in Minkowski Engine, sparse tensors have stride size and it indicates the minimum distance between neighboring elements in a dense grid the sparse tensor is defined. For example, a sparse tensor with tensor stride 8 will have at least distance 8 between all coordinates (-8, 0, 8, 16, ...).


Coordinate Manager
------------------

Many basic operations such as convolution, pooling, etc, all requires 1) hashing a set of coordinates, 2) finding neighbors between them for convolution/pooling, and 3) generating a new set of coordinates given the input coordinates. The complex operations are all managed and cached by the coordinate manager. We cache the outputs of all these operations as they are pretty expensive, and we need to reuse them very often. For example, in many conventional neural networks, there are blocks of repetitive operations such as multiple residual blocks that all shares the same architecture. Thus, recomputing all these every time is very wasteful, and we need an object that can manage and cache all of them and the coordinate manager handles all these.

In Minkowski Engine, we create a coordinate manager when :attr:`MinkowskiEngine.SparseTensor` is generated. You can optionally share an existing coordinate manager with a new :attr:`MinkowskiEngine.SparseTensor` to share the coordinates and kernel maps between them. You can access the coordinate manager of a sparse tensor by :attr:`MinkowskiEngine.SparseTensor.coords_man`.


Coordinate Key
--------------

Within a coordinate manager, all objects are cached using a hash table. A coordinate key is a hash key for coordinates of a sparse tensor. If two sparse tensors have the same coordinate manager and the same coordinate key, then the coordinates of the sparse tensors are identical and shares the same memory space and the orderings of the features are also identical.


Kernel Map
----------

A sparse tensor consists of a set of coordinates :math:`C \in \mathbb{Z}^{N \times D}` and associated features :math:`F \in \mathbb{R}^{N \times N_F}` where :math:`N` is the number of non-zero elements within a sparse tensor, :math:`D` is the dimension of the space, and :math:`N_F` is the number of channels. As we use a hash table for coordinates, the order of coordinates are arbitrary, and we need to find correct mapping between an input sparse tensor to an output sparse if we use a convolution operation.

We call this mapping from an input sparse tensor to an output sparse tensor a kernel map.  To illustrate this better, we will use a 2D convolution with kernel size 2. The convolution has :math:`2 \times 2` kernel which consists of 4 matrices, each for one of the cells of :math:`2 \times 2`. Similarly, in a sparse tensor, we require the same number of matrices to define a convolution and each cell within a convolution kernel defines a different map. We define a map as a pair of a list of integers: :math:`\mathcal{I}` and :math:`\mathcal{O}`.  A pair forms one map :math:`(\mathcal{I} \rightarrow \mathcal{O})` which defines index of input feature that maps to the index of output feature.

Since a single map is defined only on a specific cell of a convolution kernel, one convolution requires multiple maps. If we use the same :math:`2 \times 2` example, we need 4 maps to define a complete kernel map.



References
----------

- `[1] An Investigation of Sparse Tensor Formats for Tensor Libraries, 2015 <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_
