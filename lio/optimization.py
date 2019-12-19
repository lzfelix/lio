import sys
import logging
from typing import Tuple, Optional

import numpy as np
from lio.utils import ProgressBarHook

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.pso import PSO
from opytimizer.optimizers.bha import BHA
from opytimizer.spaces.hyper import HyperSpace
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils.history import History


logging.disable(sys.maxsize)


def project(z: np.ndarray, lb: float, ub: float, p: int = 2) -> np.ndarray:
    """Project a hypercomplex tensor to a real tensor in the interval [lb, ub]"""
    p_norm = np.linalg.norm(z, ord=p, axis=-1)
    factor = np.power(z.shape[-1], 1 / p)
    return lb + (ub - lb) * (p_norm / factor)


def optimize_fn(fn: callable,
                n_agents: int,
                n_vars: int,
                n_hyper_dims: int,
                n_iters: int,
                lb: float,
                ub: float,
                patience: int = 50,
                delta: float = 1e-5,
                hook_fn: Optional[callable] = None) -> History:
    def project_evaluate(z):
        return fn(project(z, lb, ub))

    lb_ = [lb] * n_vars
    ub_ = [ub] * n_vars
    space = HyperSpace(n_agents, n_vars, n_hyper_dims, n_iters, lb_, ub_)
    fun = Function(pointer=project_evaluate)

    pso = PSO()  # Using default hyperparams
    opt = Opytimizer(space, pso, fun)

    hook_fn = hook_fn or ProgressBarHook(n_iters, 'PSO')
    return opt.start(patience=patience, delta=delta, pre_evaluation_hook=hook_fn)


def finetune_projection(fn: callable,
                        z: np.ndarray,
                        fn_lb,
                        fn_ub,
                        lb=1,
                        ub=5) -> Tuple[np.ndarray, float]:
    def optimize_p(p):
        return fn(project(z, fn_lb, fn_ub, p.ravel().item()))

    space = SearchSpace(n_agents=10,
                        n_iterations=25,
                        n_variables=1,
                        lower_bound=[lb],
                        upper_bound=[ub])
    fun = Function(pointer=optimize_p)
    bha = BHA()
    opt = Opytimizer(space, bha, fun)
    history = opt.start()
    return history
