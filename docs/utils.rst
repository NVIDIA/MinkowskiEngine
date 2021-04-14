Utility Functions and Classes
=============================


sparse_quantize
---------------

.. autofunction:: MinkowskiEngine.utils.sparse_quantize


batched_coordinates
-------------------

.. autofunction:: MinkowskiEngine.utils.batched_coordinates


sparse_collate
--------------

.. autofunction:: MinkowskiEngine.utils.sparse_collate


batch_sparse_collate
--------------------

.. autofunction:: MinkowskiEngine.utils.batch_sparse_collate


cat
---

.. autofunction:: MinkowskiEngine.cat


to_sparse
---------

.. autofunction:: MinkowskiEngine.to_sparse


to_sparse_all
-------------

.. autofunction:: MinkowskiEngine.to_sparse_all


SparseCollation
---------------

.. autoclass:: MinkowskiEngine.utils.SparseCollation
    :members:
    :undoc-members:

    .. automethod:: __init__


MinkowskiToSparseTensor
-----------------------

.. autoclass:: MinkowskiEngine.MinkowskiToSparseTensor

    .. automethod:: __init__


MinkowskiToDenseTensor
-----------------------

.. autoclass:: MinkowskiEngine.MinkowskiToDenseTensor

    .. automethod:: __init__


MinkowskiToFeature
------------------

.. autoclass:: MinkowskiEngine.MinkowskiToFeature

    .. automethod:: __init__


MinkowskiStackCat
-----------------

.. autoclass:: MinkowskiEngine.MinkowskiStackCat
    :members: forward
    :undoc-members:

    .. automethod:: __init__


MinkowskiStackSum
-----------------

.. autoclass:: MinkowskiEngine.MinkowskiStackSum
    :members: forward
    :undoc-members:

    .. automethod:: __init__


MinkowskiStackMean
------------------

.. autoclass:: MinkowskiEngine.MinkowskiStackMean
    :members: forward
    :undoc-members:

    .. automethod:: __init__


MinkowskiStackVar
-----------------

.. autoclass:: MinkowskiEngine.MinkowskiStackVar
    :members: forward
    :undoc-members:

    .. automethod:: __init__