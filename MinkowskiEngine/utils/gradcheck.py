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
assert torch.__version__ >= "1.7.0", "Gradcheck requires pytorch 1.7 or higher"

from torch.types import _TensorOrTensors
from torch._six import container_abcs, istuple
import torch.testing
from torch.overrides import is_tensor_like
from itertools import product
import warnings
from typing import Callable, Union, Optional

from torch.autograd.gradcheck import _as_tuple, _differentiable_outputs, get_analytical_jacobian, get_numerical_jacobian, iter_tensors


def gradcheck(
    func: Callable[..., Union[_TensorOrTensors]],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    check_sparse_nnz: bool = False,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False
) -> bool:
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.
    The check between numerical and analytical gradients uses :func:`~torch.allclose`.
    For complex functions, no notion of Jacobian exists. Gradcheck verifies if the numerical and
    analytical values of Wirtinger and Conjugate Wirtinger derivative are consistent. The gradient
    computation is done under the assumption that the overall function has a real valued output.
    For functions with complex output, gradcheck compares the numerical and analytical gradients
    for two values of :attr:`grad_output`: 1 and 1j. For more details, check out
    :ref:`complex_autograd-doc`.
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
        check_undefined_grad (bool, options): if True, check if undefined output grads
            are supported and treated as zeros
    Returns:
        True if all differences satisfy allclose condition
    """
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    tupled_inputs = _as_tuple(inputs)
    if not check_sparse_nnz and any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)):
        return fail_test('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')

    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(
                    'The {}th input requires gradient and '
                    'is not a double precision floating point or complex. '
                    'This check will likely fail if all the inputs are '
                    'not of double precision floating point or complex. ')
            content = inp._values() if inp.is_sparse else inp
            # TODO: To cover more problematic cases, replace stride = 0 check with
            # "any overlap in memory" once we have a proper function to check it.
            if content.layout is not torch._mkldnn and \
               not all(st > 0 or sz <= 1 for st, sz in zip(content.stride(), content.size())):
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

        analytical, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian(tupled_inputs,
                                                                                                o,
                                                                                                nondet_tol=nondet_tol)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        out_is_complex = o.is_complex()

        if out_is_complex:
            # analytical vjp with grad_out = 1.0j
            analytical_with_imag_grad_out, reentrant_with_imag_grad_out, \
                correct_grad_sizes_with_imag_grad_out, correct_grad_types_with_imag_grad_out \
                = get_analytical_jacobian(tupled_inputs, o, nondet_tol=nondet_tol, grad_out=1j)
            numerical_with_imag_grad_out = get_numerical_jacobian(fn, tupled_inputs, eps=eps, grad_out=1j)

        if not correct_grad_types and check_grad_dtypes:
            return fail_test('Gradient has dtype mismatch')

        if out_is_complex and not correct_grad_types_with_imag_grad_out and check_grad_dtypes:
            return fail_test('Gradient (calculated using complex valued grad output) has dtype mismatch')

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        if out_is_complex and not correct_grad_sizes_with_imag_grad_out:
            return fail_test('Analytical gradient (calculated using complex valued grad output) has incorrect size')

        def checkIfNumericalAnalyticAreClose(a, n, j, error_str=''):
            if not torch.allclose(a, n, rtol, atol):
                return fail_test(error_str + 'Jacobian mismatch for output %d with respect to input %d,\n'
                                 'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        inp_tensors = iter_tensors(tupled_inputs, True)

        for j, (a, n, inp) in enumerate(zip(analytical, numerical, inp_tensors)):
            if a.numel() != 0 or n.numel() != 0:
                if o.is_complex():
                    # C -> C, R -> C
                    a_with_imag_grad_out = analytical_with_imag_grad_out[j]
                    n_with_imag_grad_out = numerical_with_imag_grad_out[j]
                    checkIfNumericalAnalyticAreClose(a_with_imag_grad_out, n_with_imag_grad_out, j,
                                                     "Gradients failed to compare equal for grad output = 1j. ")
                if inp.is_complex():
                    # C -> R, C -> C
                    checkIfNumericalAnalyticAreClose(a, n, j,
                                                     "Gradients failed to compare equal for grad output = 1. ")
                else:
                    # R -> R, R -> C
                    checkIfNumericalAnalyticAreClose(a, n, j)


        def not_reentrant_error(error_str=''):
            error_msg = "Backward" + error_str + " is not reentrant, i.e., running backward with same \
                        input and grad_output multiple times gives different values, \
                        although analytical gradient matches numerical gradient. \
                        The tolerance for nondeterminism was {}.".format(nondet_tol)
            return fail_test(error_msg)

        if not reentrant:
            return not_reentrant_error()

        if out_is_complex and not reentrant_with_imag_grad_out:
            return not_reentrant_error(' (calculated using complex valued grad output)')

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

        if check_undefined_grad:
            def warn_bc_breaking():
                warnings.warn((
                    'Backwards compatibility: New undefined gradient support checking '
                    'feature is enabled by default, but it may break existing callers '
                    'of this function. If this is true for you, you can call this '
                    'function with "check_undefined_grad=False" to disable the feature'))

            def check_undefined_grad_support(output_to_check):
                grads_output = [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output_to_check]
                try:
                    grads_input = torch.autograd.grad(output_to_check,
                                                      diff_input_list,
                                                      grads_output,
                                                      allow_unused=True)
                except RuntimeError:
                    warn_bc_breaking()
                    return fail_test((
                        'Expected backward function to handle undefined output grads. '
                        'Please look at "Notes about undefined output gradients" in '
                        '"tools/autograd/derivatives.yaml"'))

                for gi, i in zip(grads_input, diff_input_list):
                    if (gi is not None) and (not gi.eq(0).all()):
                        warn_bc_breaking()
                        return fail_test((
                            'Expected all input grads to be undefined or zero when all output grads are undefined '
                            'or zero. Please look at "Notes about undefined output gradients" in '
                            '"tools/autograd/derivatives.yaml"'))
                return True

            # All backward functions must work properly if all output grads are undefined
            outputs_to_check = [[torch._C._functions.UndefinedGrad()(o) for o in _differentiable_outputs(func.apply(*tupled_inputs))]]

            # If there are multiple output grads, we should be able to undef one at a time without error
            if len(outputs_to_check[0]) > 1:
                for undef_grad_idx in range(len(output)):
                    output_to_check = _differentiable_outputs(func.apply(*tupled_inputs))
                    outputs_to_check.append([
                        torch._C._functions.UndefinedGrad()(o) if idx == undef_grad_idx else o
                        for idx, o in enumerate(output_to_check)])

            for output_to_check in outputs_to_check:
                if not check_undefined_grad_support(output_to_check):
                    return False

    return True
