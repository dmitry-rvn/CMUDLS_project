"""
Optimization module
"""
import numpy as np

import needle as ndl


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            try:
                u_prev = self.u[i]
            except KeyError:
                u_prev = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype)
                self.u[i] = u_prev

            grad = p.grad.data + self.weight_decay * p.data
            self.u[i] = self.momentum * u_prev + (1 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[i]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            if i not in self.m:
                self.m[i] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype)
                self.v[i] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype)

            grad = p.grad.data + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_ = self.m[i] / (1 - self.beta1 ** self.t)
            v_ = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_ / (v_ ** 0.5 + self.eps)
