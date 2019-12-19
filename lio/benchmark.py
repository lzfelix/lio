# I swear I've tested these
from typing import NamedTuple, Callable
import numpy as np


def _coerce(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        return np.asarray(x, np.float64)
    return x


def sphere(x: np.ndarray) -> float:
    x = _coerce(x)
    return np.sum(np.power(x, 2)).item()


def csendes(x: np.ndarray, eps: float=1e-20) -> float:
    x = _coerce(x)
    return np.sum(np.power(x, 6) * (2 + np.sin(1 / (x + eps)))).item()


def salomon(x: np.ndarray) -> float:
    x = _coerce(x)
    w = np.sqrt(np.sum(np.power(x, 2)))
    return 1 - np.cos(2 * np.pi * w) + 0.1 * w


def ackley_first(x: np.ndarray) -> float:
    x = _coerce(x)
    inv_dim = 1 / x.shape[0]
    term_1 = -0.02 * np.sqrt(inv_dim * np.sum(np.power(x, 2)))
    term_2 = inv_dim * np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(term_1) - np.exp(term_2) + 20 + np.e


def alpine_first(x: np.ndarray) -> float:
    x = _coerce(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x)).item()


def rastrigin(x: np.ndarray) -> float:
    x = _coerce(x)
    dim = x.shape[0]
    cosine = 10 * np.cos(2 * np.pi * x)
    return 10 * dim + np.sum(np.power(x, 2) - cosine)


def schwefel(x: np.ndarray) -> float:
    x = _coerce(x)
    alpha = np.sqrt(np.pi)  # Using LibOPT constant
    inner = np.sum(np.power(x, 2))
    return np.power(inner, alpha)

def brown(x: np.ndarray) -> float:
    """Note: This function always returns 0 when x is a single scalar."""
    x = _coerce(x)
    squares_left = np.power(x[:-1], 2)
    squares_right = np.power(x[1:], 2)

    inner = np.power(squares_left, squares_right + 1) + np.power(squares_right, squares_left + 1)
    return np.sum(inner).item()


class BnFn(NamedTuple):
    function: Callable[[np.ndarray], float]
    lb: float
    ub: float


def get_function(fun_name: str) -> BnFn:
    return {
        'sphere': BnFn(sphere, -10, 10),
        'csendes': BnFn(csendes, -1, 1),
        'salomon': BnFn(salomon, -100, 100),
        'ackley1': BnFn(ackley_first, -35, 35),
        'alpine1': BnFn(alpine_first, -10, 10),
        'rastrigin': BnFn(rastrigin, -5.12, 5.12),
        'schwefel': BnFn(schwefel, -100, 100),
        'brown': BnFn(brown, -1, 4)
    }[fun_name]
