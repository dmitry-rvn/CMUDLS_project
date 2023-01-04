"""
The module.
"""
from typing import List

from needle import Tensor, ops, init


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype=dtype, device=device, requires_grad=True))
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, dtype=dtype, device=device, requires_grad=True)
            self.bias = Parameter(ops.transpose(self.bias))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bias:
            tmp = ops.matmul(x, self.weight)
            return tmp + ops.broadcast_to(self.bias, shape=tmp.shape)
        return ops.matmul(x, self.weight)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for m in self.modules:
            x = m(x, *args, **kwargs)
        return x


class SoftmaxLoss(Module):
    """
    SoftmaxLoss with `mean` reduction.
    """
    def forward(self, logits: Tensor, y: Tensor):
        lse = ops.logsumexp(logits, axes=(1,))
        z_y = ops.summation(logits * init.one_hot(logits.shape[1], y, device=logits.device), axes=(1,))
        return ops.summation(lse - z_y) / logits.shape[0]


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        e_x = ops.reshape(ops.summation(x, axes=(1,)), shape=(batch_size, 1)) / x.shape[1]
        numerator = x - ops.broadcast_to(e_x, x.shape)

        var_x = ops.reshape(ops.summation(numerator ** 2, axes=(1,)), shape=(batch_size, 1)) / x.shape[1]
        denominator = (var_x + self.eps) ** 0.5
        denominator = ops.broadcast_to(denominator, x.shape)

        return ops.multiply(ops.broadcast_to(self.weight.reshape((1, self.dim)), x.shape), numerator / denominator) + ops.broadcast_to(self.bias.reshape((1, self.dim)), x.shape)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return (x * init.randb(*x.shape, p=1 - self.p, device=x.device)) / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, batch_first: bool = False, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.
        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector
        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.batch_first = batch_first
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors
        Input:
        x of shape (seq_len, bs)
        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        if self.batch_first:
            x = x.transpose((1, 0))

        one_hot_vectors = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)

        seq_len, bs, em = one_hot_vectors.shape

        one_hot_vectors = one_hot_vectors.reshape((seq_len * bs, em))
        out = one_hot_vectors @ self.weight

        out = out.reshape((seq_len, bs, self.embedding_dim))

        if self.batch_first:
            out = out.transpose((1, 0))
        return out
