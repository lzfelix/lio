import sys
import logging
import argparse
import multiprocessing as mp
from typing import List

import numpy as np
import tqdm.auto as tqdm

sys.path.append('../')
from lio import benchmark
from lio import optimization
from lio.utils import QueueProgressBarHook

logging.disable(sys.maxsize)
N_HYPER_DIMS = 4
PATIENCE = 50
PATIENCE_DELTA = 1e-5


def get_exec_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-fn', type=str, help='Benchmark function')
    parser.add_argument('-n_agents', type=int, default=100)
    parser.add_argument('-n_vars', type=int, default=10)
    parser.add_argument('-n_iters', type=int, default=0, help='If 0, uses 2000*n_vars')
    parser.add_argument('-n_runs', type=int, default=15)
    args = parser.parse_args()

    if args.n_iters <= 0:
        args.n_iters = 2000 * args.n_vars

    return args


def run_experiment(hook: QueueProgressBarHook, bnfn: benchmark.BnFn, args: argparse.Namespace) -> dict:
    # Avoid child process inheriting seed from parent
    np.random.seed()

    history = optimization.optimize_fn(
        bnfn.function,
        n_agents=args.n_agents,
        n_vars=args.n_vars,
        n_hyper_dims=N_HYPER_DIMS,
        n_iters=args.n_iters,
        lb=bnfn.lb,
        ub=bnfn.ub,
        patience=PATIENCE,
        delta=PATIENCE_DELTA,
        hook_fn=hook
    )

    # The best agent location
    z = np.asarray(history.best_agent[-1][0])
    finetune_history = optimization.finetune_projection(bnfn.function, z, bnfn.lb, bnfn.ub)

    finetuned_p = np.asarray(finetune_history.best_agent[-1][0]).ravel().item()
    finetuned_fitness = finetune_history.best_agent[-1][1]
    hook.notify_with_p(history.best_agent[-1][1], finetuned_fitness, finetuned_p)
    hook.finish()
    return dict(opt_history=history, lio_history=finetune_history)


def create_pbars(n_bars: int, total: int) -> List[tqdm.tqdm]:
    bars = []
    for i in range(n_bars):
        # +2 because 1 is used for LIO and another for agents initialization
        bar = tqdm.tqdm(total=total + 2, position=i)
        bar.set_description(f'Task {i + 1}')
        bars.append(bar)
    return bars


if __name__ == '__main__':
    exec_args = get_exec_args()
    print(exec_args)
    sys.stdout.flush()

    n_processes = min(mp.cpu_count(), exec_args.n_runs)
    pool = mp.Pool(n_processes)
    queue = mp.Manager().Queue()

    try:
        bnfn = benchmark.get_function(exec_args.fn)
    except KeyError:
        raise ValueError('Invalid benchmark function name')

    hooks = [QueueProgressBarHook(task_id, queue) for task_id in range(exec_args.n_runs)]
    pbars = create_pbars(exec_args.n_runs, exec_args.n_iters)
    arguments = [(hook, bnfn, exec_args) for hook, pbar in zip(hooks, pbars)]

    finished = 0
    results = pool.starmap_async(run_experiment, arguments)

    while True:
        msg = queue.get()
        if msg.fitness:
            pbars[msg.task_id].update(n=1)
            pbars[msg.task_id].set_postfix({
                'f(z, 2)': msg.fitness,
                'p*': msg.fine_p or '?',
                'f(z, p*)': msg.fine_fitness or '?'
            })
        else:
            pbars[msg.task_id].close()
            finished += 1
            if finished == exec_args.n_runs:
                break

    pool.close()
    pool.join()

    results = results.get()