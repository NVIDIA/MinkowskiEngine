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

from torch.autograd.gradcheck import _as_tuple, _differentiable_outputs, get_analytical_jacobian, get_numerical_jacobian, iter_tensors


def gradcheck(func,
              inputs,
              eps=1e-6,
              atol=1e-5,
              rtol=1e-3,
              raise_exception=True):
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients has the same behaviour as
    `numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_,
    i.e., it checks that

    .. math::

        \lvert a - n \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert n \rvert

    holds for all elements of analytical gradient :math:`a` and numerical
    gradient :math:`n`.

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
        inputs (tuple of Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.

    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    # Make sure that gradients are saved for all inputs
    any_input_requiring_grad = False
    for inp in tupled_inputs:
        if isinstance(inp, torch.Tensor):
            if inp.requires_grad:
                if inp.dtype != torch.float64:
                    warnings.warn(
                        'At least one of the inputs that requires gradient '
                        'is not of double precision floating point. '
                        'This check will likely fail if all the inputs are '
                        'not of double precision floating point. ')
                any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')

    output = _differentiable_outputs(func.apply(*inputs))

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func.apply(*input))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(
            tupled_inputs, o)
        numerical = get_numerical_jacobian(fn, inputs, eps=eps)

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                succ_index = (a - n).abs() <= (atol + rtol * n.abs())
                if not succ_index.all():
                    return fail_test(
                        'Jacobian mismatch for output %d with respect to input %d,\n'
                        'numerical:%s\nanalytical:%s\nsuccess:%s\ndifference a - n:%s\n'
                        % (i, j, n, a, succ_index, a - n))

        if not reentrant:
            return fail_test(
                'Backward is not reentrant, i.e., running backward with same '
                'input and grad_output multiple times gives different values, '
                'although analytical gradient matches numerical gradient')

    # check if the backward multiplies by grad_output
    output = _differentiable_outputs(func.apply(*inputs))
    if any([o.requires_grad for o in output]):
        diff_input_list = list(iter_tensors(inputs, True))
        if not diff_input_list:
            raise RuntimeError("no Tensors requiring grad found in input")
        grads_input = torch.autograd.grad(
            output,
            diff_input_list, [torch.zeros_like(o) for o in output],
            allow_unused=True)
        for gi, i in zip(grads_input, diff_input_list):
            if gi is None:
                continue
            if not gi.eq(0).all():
                return fail_test('backward not multiplied by grad_output')
            if gi.type() != i.type():
                return fail_test("grad is incorrect type")
            if gi.size() != i.size():
                return fail_test('grad is incorrect size')

    return True
