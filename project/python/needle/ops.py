"""
Operator table.
"""
import math
from typing import Optional, Tuple

import numpy

from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def __repr__(self):
        return f'AddScalar: scalar={self.scalar}'

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def __repr__(self):
        return f'MulScalar: scalar={self.scalar}'

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def __repr__(self):
        return f'PowerScalar: scalar={self.scalar}'

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        x = node.inputs[0]
        return out_grad * self.scalar * (x ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = node.inputs
        return out_grad * (y ** -1), out_grad * -x * (y ** -2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def __repr__(self):
        return f'DivScalar: scalar={self.scalar}'

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def __repr__(self):
        return f'Transpose: axes={self.axes}'

    def compute(self, a: NDArray) -> NDArray:
        order = list(range(len(a.shape)))
        axes = self.axes if self.axes is not None else (order[-1], order[-2])
        order[axes[0]] = axes[1]
        order[axes[1]] = axes[0]
        return a.permute(tuple(order))

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return transpose(out_grad, self.axes)


def transpose(a, axes: Optional[tuple] = None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape: tuple):
        self.shape = shape

    def __repr__(self):
        return f'Reshape: shape={self.shape}'

    def compute(self, a: NDArray) -> NDArray:
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape: tuple):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape: tuple):
        self.shape = shape

    def __repr__(self):
        return f'BroadcastTo: shape={self.shape}'

    def compute(self, a: NDArray) -> NDArray:
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        x = node.inputs[0]
        axes = []
        shape = [1] * (len(self.shape) - len(x.shape)) + list(x.shape)
        for i, s in enumerate(self.shape):
            if i >= len(shape) or s != shape[i]:
                axes.append(i)
        return reshape(summation(out_grad, tuple(axes)), x.shape)


def broadcast_to(a, shape: tuple):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes=None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def __repr__(self):
        return f'Summation: axes={self.axes}'

    def compute(self, a: NDArray) -> NDArray:
        if self.axes is None:
            return a.sum()
        for i, axis in enumerate(sorted(self.axes)):
            a = a.sum(axis - i)
        return a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        x = node.inputs[0]

        shape_out = [1] * len(x.shape)
        axes = set(self.axes) if self.axes is not None else set(range(len(x.shape)))

        grad_i = 0
        for x_i in range(len(x.shape)):
            if x_i not in axes:
                shape_out[x_i] = out_grad.shape[grad_i]
                grad_i += 1

        return broadcast_to(reshape(out_grad, tuple(shape_out)), x.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = node.inputs
        d_x = out_grad.matmul(y.transpose())
        d_y = x.transpose().matmul(out_grad)
        if d_y.shape != y.shape:
            sum_by_axes = tuple(range(len(d_y.shape) - len(y.shape)))
            d_y = d_y.sum(sum_by_axes)
        if d_x.shape != x.shape:
            sum_by_axes = tuple(range(len(d_x.shape) - len(x.shape)))
            d_x = d_x.sum(sum_by_axes)
        return d_x, d_y


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a * (-1)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * (-1)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a.exp()

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        return out_grad * mask


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def __repr__(self):
        return f'LogSumExp: axes={self.axes}'

    def compute(self, a: NDArray) -> NDArray:
        a_max = a.max(self.axes, keepdims=True)
        lse = array_api.log(array_api.exp(a - a_max.broadcast_to(a.shape)).sum(axis=self.axes, keepdims=True)) + a_max

        if self.axes is None:
            shape_out = [1]
        else:
            axes = list(self.axes)
            shape_out = [size for i, size in enumerate(a.shape) if i not in axes]

        return lse.reshape(tuple(shape_out))

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        x = node.inputs[0]

        if self.axes is None:
            return out_grad.broadcast_to(x.shape) * exp(x - node)

        shape = [1] * len(x.shape)
        axes = set(self.axes)

        j = 0
        for i in range(len(shape)):
            if i not in axes:
                shape[i] = node.shape[j]
                j += 1

        return out_grad.reshape(shape).broadcast_to(x.shape) * exp(x - node.reshape(shape).broadcast_to(x.shape))


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def __repr__(self):
        return f'Stack: axis={self.axis}'

    def compute(self, args):
        input_shape = args[0].shape

        new_shape = list(input_shape)
        new_shape.insert(self.axis, len(args))
        out = array_api.empty(new_shape, dtype=args[0].dtype, device=args[0].device)

        slices = []
        for i, s in enumerate(new_shape):
            slices.append(slice(s) if i != self.axis else 0)

        for i, a in enumerate(args):
            slices[self.axis] = i
            out[tuple(slices)] = a.reshape((1,) + input_shape)
        return out

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis: int):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def __repr__(self):
        return f'Split: axis={self.axis}'

    def compute(self, a):
        slices = [slice(0, s, 1) if i != self.axis else 0 for i, s in enumerate(a.shape)]
        tensors = []
        new_shape = tuple(s for i, s in enumerate(a.shape) if i != self.axis)
        for i in range(a.shape[self.axis]):
            slices[self.axis] = i
            tensors.append(a[tuple(slices)].reshape(new_shape))
        return tuple(tensors)

    def gradient(self, out_grad, node):
        return stack(tuple(out_grad), self.axis)


def split(a, axis: int):
    return Split(axis)(a)


def softmax(x: Tensor) -> Tensor:
    x_max = Tensor(x.realize_cached_data().max(axis=len(x.shape) - 1, keepdims=True).broadcast_to(x.shape), device=x.device, requires_grad=False)

    z = exp(x - x_max)
    z_sum = summation(z, axes=len(x.shape) - 1)
    z_sum = z_sum.reshape((*x.shape[:-1], 1))
    z_sum = broadcast_to(z_sum, x.shape)
    return z / z_sum


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """
    Batch matrix multiplication.
    Example: bmm(a[16 x 256 x 128 x 128], b[16 x 256 x 128 x 64]) => c[16 x 256 x 128 x 64]
    """
    assert len(a.shape) == len(b.shape), f'{len(a.shape)} != {len(b.shape)}'
    batch_dims = a.shape[:len(a.shape) - 2]
    assert batch_dims == b.shape[:len(b.shape) - 2], f'{batch_dims} != {b.shape[:len(b.shape) - 2]}'

    n = math.prod(batch_dims)
    a = a.reshape((n, *a.shape[-2:]))
    b = b.reshape((n, *b.shape[-2:]))
    out = stack([a_ @ b_ for a_, b_ in zip(split(a, axis=0), split(b, axis=0))], axis=0)
    out = out.reshape((*batch_dims, *out.shape[-2:]))
    return out
