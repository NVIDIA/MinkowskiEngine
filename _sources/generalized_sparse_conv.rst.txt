Generalized Sparse Convolution
==============================

Sparse Tensor
-------------

The :attr:`SparseTensor` class is the basic tensor in MinkowskiEngine. We use
the COOrdinate (COO) format to save a sparse tensor `[1]
<http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_. This
representation is simply a concatenation of coordinates in a matrix :math:`C`
and associated features :math:`F`.

.. math::

   \mathbf{C} = \begin{bmatrix}
   x_1^1   & x_1^2  & \cdots & x_1^D  & b_1    \\
    \vdots & \vdots & \ddots & \vdots & \vdots \\
   x_N^1   & x_N^2  & \cdots & x_N^D  & b_N
   \end{bmatrix}, \; \mathbf{F} = \begin{bmatrix}
   \mathbf{f}_1^T\\
   \vdots\\
   \mathbf{f}_N^T
   \end{bmatrix}

In the above equation, we use a :math:`D`-dimensional space and :math:`N`
number of points, each with the coordinate :math:`(x_i^1, x_i^1, \cdots,
x_i^D)`, and the associated feature :math:`\mathbf{f}_i`. :math:`b_i` indicates
the mini-batch index to disassociate instances within the same batch.
Internally, we handle the batch index as an additional spatial dimension.


Generalized Sparse Convolution
------------------------------

The convolution is a fundamental operation in many fields. In image perception,
the 2D convolution has achieved state-of-the-art performance in many tasks and
is proven to be the most crucial operation in AI, and computer vision research.

In this section, we generalize the sparse convolution for generic input and
output coordinates, and for arbitrary kernel shapes. The generalized sparse
convolution encompasses not only all sparse convolutions but also the
conventional dense convolutions. Let :math:`x^{\text{in}}_\mathbf{u} \in
\mathbb{R}^{N^\text{in}}` be an :math:`N^\text{in}$`-dimensional input feature
vector in a :math:`D`-dimensional space at :math:`\mathbf{u} \in \mathbb{R}^D`
(a D-dimensional coordinate), and convolution kernel weights be
:math:`\mathbf{W} \in \mathbb{R}^{K^D \times N^\text{out} \times N^\text{in}}`.
We break down the weights into spatial weights with :math:`K^D` matrices of
size :math:`N^\text{out} \times N^\text{in}` as :math:`W_\mathbf{i}` for
:math:`|\{\mathbf{i}\}_\mathbf{i}| = K^D`. Then, the conventional dense
convolution in D-dimension is

.. math::
   \mathbf{x}^{\text{out}}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{V}^D(K)} W_\mathbf{i} \mathbf{x}^{\text{in}}_{\mathbf{u} + \mathbf{i}} \text{ for } \mathbf{u} \in \mathbb{Z}^D,

where :math:`\mathcal{V}^D(K)` is the list of offsets in :math:`D`-dimensional
hypercube centered at the origin. e.g. :math:`\mathcal{V}^1(3)=\{-1, 0, 1\}`.
The generalized sparse convolution in the following equation relaxes the above
equation.

.. math::
   \mathbf{x}^{\text{out}}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u}, \mathcal{C}^{\text{in}})} W_\mathbf{i} \mathbf{x}^{\text{in}}_{\mathbf{u} + \mathbf{i}} \text{ for } \mathbf{u} \in \mathcal{C}^{\text{out}}

where math:`\mathcal{N}^D` is a set of offsets that define the shape of a
kernel and :math:`\mathcal{N}^D(\mathbf{u}, \mathcal{C}^\text{in})=
\{\mathbf{i} | \mathbf{u} + \mathbf{i} \in \mathcal{C}^\text{in}, \mathbf{i}
\in \mathcal{N}^D \}` as the set of offsets from the current center,
:math:`\mathbf{u}`, that exist in :math:`\mathcal{C}^\text{in}`.
:math:`\mathcal{C}^\text{in}` and :math:`\mathcal{C}^\text{out}` are predefined
input and output coordinates of sparse tensors. First, note that the input
coordinates and output coordinates are not necessarily the same.  Second, we
define the shape of the convolution kernel arbitrarily with
:math:`\mathcal{N}^D`. This generalization encompasses many special cases such
as the dilated convolution and typical hypercubic kernels. Another interesting
special case is the sparse submanifold convolution when we set
:math:`\mathcal{C}^\text{out} = \mathcal{C}^\text{in}` and :math:`\mathcal{N}^D
= \mathcal{V}^D(K)`. If we set :math:`\mathcal{C}^\text{in} =
\mathcal{C}^\text{out} = \mathbb{Z}^D` and :math:`\mathcal{N}^D =
\mathcal{V}^D(K)`, the generalized sparse convolution becomes the conventional
dense convolution.  If we define the :math:`\mathcal{C}^\text{in}` and
:math:`\mathcal{C}^\text{out}` as multiples of a natural number and
:math:`\mathcal{N}^D = \mathcal{V}^D(K)`, we have a strided dense convolution.


References
----------

- `[1] An Investigation of Sparse Tensor Formats for Tensor Libraries, 2015 <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_
- `[2] 3D Semantic Segmentation with Submanifold Sparse Convolutional Neural Networks, CVPR'18 <https://arxiv.org/abs/1711.10275>`_
- `[3] 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19 <https://arxiv.org/abs/1904.08755>`_
