# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.testing
import warnings
from typing import Callable, Union, Optional

from torch.autograd.gradcheck import _as_tuple, _differentiable_outputs, get_analytical_jacobian, get_numerical_jacobian, iter_tensors


def gradcheck(
    func,
    inputs,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    check_sparse_nnz: bool = False,
    nondet_tol: float = 0.0
) -> bool:
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.
    The check between numerical and analytical gradients uses :func:`~torch.allclose`.
    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.
    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.
    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
            and for any SparseTensor at input, gradcheck will perform check at nnz positions only.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
    Returns:
        True if all differences satisfy allclose condition
    """
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    tupled_inputs = _as_tuple(inputs)
    if any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)) and not check_sparse_nnz:
        return fail_test('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')

    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(
                    'The {}th input requires gradient and '
                    'is not a double precision floating point or complex. '
                    'This check will likely fail if all the inputs are '
                    'not of double precision floating point or complex. ')
            content = inp._values() if inp.is_sparse else inp
            if content.layout is not torch._mkldnn and any([s == 0 for s in content.stride()]):
                raise RuntimeError(
                    'The {}th input has a dimension with stride 0. gradcheck only '
                    'supports inputs that are non-overlapping to be able to '
                    'compute the numerical gradients correctly. You should call '
                    '.contiguous on the input before passing it to gradcheck.')
            any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')

    func_out = func.apply(*tupled_inputs)
    output = _differentiable_outputs(func_out)

    if not output:
        for i, o in enumerate(func_out):
            def fn(input):
                return _as_tuple(func.apply(*input))[i]
            numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)
            for n in numerical:
                if torch.ne(n, 0).sum() > 0:
                    return fail_test('Numerical gradient for function expected to be zero')
        return True

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func.apply(*input))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o, nondet_tol=nondet_tol)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                if not torch.allclose(a, n, rtol, atol):
                    return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
                                     'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        if not reentrant:
            return fail_test('Backward is not reentrant, i.e., running backward with same '
                             'input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient. '
                             'The tolerance for nondeterminism was {}.'.format(nondet_tol))

    # check if the backward multiplies by grad_output
    output = _differentiable_outputs(func.apply(*tupled_inputs))
    if any([o.requires_grad for o in output]):
        diff_input_list = list(iter_tensors(tupled_inputs, True))
        if not diff_input_list:
            raise RuntimeError("no Tensors requiring grad found in input")
        grads_input = torch.autograd.grad(output, diff_input_list,
                                          [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output],
                                          allow_unused=True)
        for gi, i in zip(grads_input, diff_input_list):
            if gi is None:
                continue
            if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
                if gi.layout != i.layout:
                    return fail_test('grad is incorrect layout (' + str(gi.layout) + ' is not ' + str(i.layout) + ')')
                if gi.layout == torch.sparse_coo:
                    if gi.sparse_dim() != i.sparse_dim():
                        return fail_test('grad is sparse tensor, but has incorrect sparse_dim')
                    if gi.dense_dim() != i.dense_dim():
                        return fail_test('grad is sparse tensor, but has incorrect dense_dim')
                gi = gi.to_dense()
                i = i.to_dense()
            if not gi.eq(0).all():
                return fail_test('backward not multiplied by grad_output')
            if gi.dtype != i.dtype or gi.device != i.device or gi.is_sparse != i.is_sparse:
                return fail_test("grad is incorrect type")
            if gi.size() != i.size():
                return fail_test('grad is incorrect size')

    return True
