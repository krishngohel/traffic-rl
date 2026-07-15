"""A small, fully seeded double-DQN in pure NumPy.

The network is 22 -> 64 -> 64 -> 2; at that scale NumPy trains it in minutes and
the whole agent stays transparent, dependency-light, and bit-reproducible —
which matters more here than GPU throughput. Standard pieces: replay buffer,
target network, Adam, Huber loss, epsilon-greedy over *legal* actions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class MLP:
    """Two-hidden-layer ReLU network with He initialization."""

    def __init__(self, sizes: tuple[int, ...], rng: np.random.Generator):
        self.params: list[np.ndarray] = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=False):
            w = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
            self.params += [w.astype(np.float64), np.zeros(n_out)]

    def forward(self, x: np.ndarray, cache: list | None = None) -> np.ndarray:
        h = np.atleast_2d(x)
        n_layers = len(self.params) // 2
        for layer in range(n_layers):
            w, b = self.params[2 * layer], self.params[2 * layer + 1]
            z = h @ w + b
            if layer < n_layers - 1:
                if cache is not None:
                    cache.append((h, z))
                h = np.maximum(z, 0.0)
            else:
                if cache is not None:
                    cache.append((h, z))
                h = z
        return h

    def backward(self, cache: list, dout: np.ndarray) -> list[np.ndarray]:
        """Gradients for all params given d(loss)/d(output). Mirrors forward()."""
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        n_layers = len(self.params) // 2
        grad = dout
        for layer in reversed(range(n_layers)):
            h_in, z = cache[layer]
            if layer < n_layers - 1:
                grad = grad * (z > 0.0)
            w = self.params[2 * layer]
            grads[2 * layer] = h_in.T @ grad
            grads[2 * layer + 1] = grad.sum(axis=0)
            grad = grad @ w.T
        return grads

    def copy_from(self, other: MLP) -> None:
        self.params = [p.copy() for p in other.params]

    def save(self, path) -> None:
        np.savez_compressed(path, *self.params)

    @classmethod
    def load(cls, path) -> MLP:
        data = np.load(path)
        net = cls.__new__(cls)
        net.params = [data[k] for k in sorted(data.files, key=lambda s: int(s.split("_")[1]))]
        return net


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 3e-4):
        self.lr = lr
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        bias1 = 1.0 - self.b1**self.t
        bias2 = 1.0 - self.b2**self.t
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p -= self.lr * (self.m[i] / bias1) / (np.sqrt(self.v[i] / bias2) + self.eps)


@dataclass
class Replay:
    capacity: int
    n_features: int
    _n: int = 0
    _i: int = 0
    s: np.ndarray = field(init=False)
    a: np.ndarray = field(init=False)
    r: np.ndarray = field(init=False)
    s2: np.ndarray = field(init=False)
    mask2: np.ndarray = field(init=False)
    done: np.ndarray = field(init=False)

    def __post_init__(self):
        self.s = np.zeros((self.capacity, self.n_features), dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.s2 = np.zeros((self.capacity, self.n_features), dtype=np.float32)
        self.mask2 = np.zeros((self.capacity, 2), dtype=bool)
        self.done = np.zeros(self.capacity, dtype=bool)

    def add(self, s, a, r, s2, mask2, done) -> None:
        i = self._i
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.mask2[i], self.done[i] = s2, mask2, done
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self._n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.mask2[idx], self.done[idx]

    def __len__(self) -> int:
        return self._n


class DQN:
    def __init__(
        self,
        n_features: int,
        n_actions: int = 2,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        sizes = (n_features, hidden, hidden, n_actions)
        self.online = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.online)
        self.opt = Adam(self.online.params, lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions

    def act(self, features: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            legal = np.flatnonzero(mask)
            return int(self.rng.choice(legal))
        return self.greedy(features, mask)

    def greedy(self, features: np.ndarray, mask: np.ndarray) -> int:
        q = self.online.forward(features)[0]
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def train_step(self, replay: Replay, batch: int = 64) -> float:
        s, a, r, s2, mask2, done = replay.sample(batch, self.rng)
        # Double DQN: online net picks the next action (legal only), target net
        # evaluates it.
        q2_online = self.online.forward(s2)
        q2_online = np.where(mask2, q2_online, -np.inf)
        a2 = np.argmax(q2_online, axis=1)
        q2_target = self.target.forward(s2)
        target = r + (~done) * self.gamma * q2_target[np.arange(batch), a2]

        cache: list = []
        q = self.online.forward(s, cache)
        td = q[np.arange(batch), a] - target
        # Huber (delta = 1) gradient, only through the taken action.
        dq = np.zeros_like(q)
        dq[np.arange(batch), a] = np.clip(td, -1.0, 1.0) / batch
        grads = self.online.backward(cache, dq)
        self.opt.step(self.online.params, grads)
        return float(np.mean(np.minimum(0.5 * td**2, np.abs(td) - 0.5)))

    def sync_target(self) -> None:
        self.target.copy_from(self.online)
