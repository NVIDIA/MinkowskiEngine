Definitions and Terminology
===========================

Sparse Tensor
-------------

A sparse tensor is a high-dimensional extension of a sparse matrix where non-zero elements are represented as a set of indices :math:`\mathcal{C}` and associated values (or features) :math:`\mathcal{F}`. We use the COOrdinate list (COO) format to save a sparse tensor `[1] <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_. This representation is simply a concatenation of coordinates into a matrix :math:`C` and associated values or features :math:`F`. In traditional sparse tensor, indices or coordinates have to be non-negative integers, whereas, in Minkowski Engine, negative coordinates are also valid coordinates. A final sparse tensor :math:`\mathscr{T}` with :math:`D` dimensional coordinates is a rank-:math:`D` tensor if features are scalars, or a rank-:math:`D + 1` if features are vectors.

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

.. math::

   \mathscr{T}[x^1_i,  x^2_i,  \cdots, x^D_i] = \begin{cases}
      \mathbf{f}_i \;\; & \text{if} \; (x^1_i,  x^2_i, \cdots, x^D_i) \in \mathcal{C} \\
      0   \;\; & \text{otherwise}
   \end{cases}

In sum, a sparse tensor consists of a set of coordinates :math:`\mathcal{C}` or equivalently a coordinate matrix :math:`C \in \mathbb{Z}^{N \times D}` and associated features :math:`\mathcal{F}` or a feature matrix :math:`F \in \mathbb{R}^{N \times N_F}` where :math:`N` is the number of non-zero elements within a sparse tensor, :math:`D` is the dimension of the space, and :math:`N_F` is the number of channels. The rest of the elements in a sparse tensor is 0.


Tensor Stride
-------------

In 2D ConvNets, strided convolution/pooling layers (stride > 1) creates a new feature map with lower resolutions. The strided convolution/pooling allows neural networks to have exponentially large receptive field size by shrinking down the space a convolution/pooling layer operates on. For example, if we apply a stride 2 convolution on a 32 :math:`\times` 32 pixel image, the output feature map has half the resolution of the input, 16 :math:`\times` 16 . The pixels in the 16 :math:`\times` 16 resolution image, however, have distance 2 between adjacent pixels. Similarly, we can define the tensor stride using the dense tensor a sparse tensor is derived from, which indicates the minimum distance between adjacent cells in the original dense grid the sparse tensor is defined. For example, a sparse tensor with tensor stride 8 will have at least distance 8 between all coordinates (-8, 0, 8, 16, ...).


Coordinate Manager
------------------

Many basic operations such as convolution, pooling, etc, all require finding neighbors between non-zero elements, or generating a new set of coordinates given the input coordinates. All these operations are all managed and cached by the coordinate manager. We cache the outputs of each call as it is quite time-consuming, and we reuse them very often. For example, in many conventional neural networks, there are blocks of repetitive operations such as multiple residual blocks all of which have the same convolution.

In Minkowski Engine, we create a coordinate manager when :attr:`MinkowskiEngine.SparseTensor` is called. You can optionally share an existing coordinate manager with a new :attr:`MinkowskiEngine.SparseTensor` to share the coordinate manager and the kernel maps between them. You can access the coordinate manager of a sparse tensor with :attr:`MinkowskiEngine.SparseTensor.coords_man`.


Coordinate Key
--------------

Within a coordinate manager, all objects are cached using a map. A coordinate key is a hash key for the map that saves the coordinates of sparse tensor(s). If two sparse tensors have the same coordinate manager and the same coordinate key, then the coordinates of the sparse tensors are identical and they share the same memory space.


Kernel Map
----------

A sparse tensor consists of a set of coordinates :math:`C \in \mathbb{Z}^{N \times D}` and associated features :math:`F \in \mathbb{R}^{N \times N_F}` where :math:`N` is the number of non-zero elements within a sparse tensor, :math:`D` is the dimension of the space, and :math:`N_F` is the number of channels. As we use a hash table for coordinates, the order of coordinates is arbitrary, and we need to find the correct mapping between an input sparse tensor to an output sparse if we use a convolution operation.

We call this mapping from an input sparse tensor to an output sparse tensor a kernel map.  For example, a 2D convolution with kernel size 3 has a :math:`3 \times 3` convolution kernel, which consists of 9 matrices. With each kernel, some input coordinates will be mapped to corresponding output coordinates. We define such maps as pairs of lists of integers: :math:`\mathcal{I}` and :math:`\mathcal{O}`.  A pair forms one map :math:`(\mathcal{I} \rightarrow \mathcal{O})` which defines row indices of input feature :math:`F_I` that map to the row indices of output feature :math:`F_O`.

Since a single map is defined only on a specific cell of a convolution kernel, one convolution requires multiple maps. In the case of a :math:`3 \times 3` convolution, we need 9 maps to define a complete kernel map.


.. |kernel_map| image:: images/kernel_map.gif
   :width: 100%

+-------------------------------------------------------+
| Convolution Kernel Map                                |
+-------------------------------------------------------+
| |kernel_map|                                          |
+-------------------------------------------------------+
| [Photo Credit: `Chris Choy <https://chrischoy.org>`_] |
+-------------------------------------------------------+

Using the above example, we can extract the convolution kernel map as follows:

- Kernel B: 1 → 0
- Kernel B: 0 → 2
- Kernel H: 2 → 3
- Kernel I: 0 → 2


References
----------

- `[1] An Investigation of Sparse Tensor Formats for Tensor Libraries, 2015 <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_
