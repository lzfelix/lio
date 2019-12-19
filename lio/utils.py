from typing import NamedTuple, Optional

from tqdm import auto as tqdm


class ProgressBarHook:
    def __init__(self, total, description):
        self.pbar = tqdm.tqdm(total=total)
        if description:
            self.pbar.set_description(description)

    def __call__(self, mh, space, fn):
        self.pbar.update(n=1)
        self.pbar.set_postfix(dict(fitness=space.best_agent.fit))
        if self.pbar.n == self.pbar.total:
            self.pbar.close()


class QueueMessage(NamedTuple):
    task_id: int
    fitness: Optional[float]
    fine_fitness: Optional[float]
    fine_p: Optional[float]


class QueueProgressBarHook:
    def __init__(self, task_id, queue):
        self.task_id = task_id
        self.queue = queue

    def notify_with_p(self, fitness, lio_fitness, lio_p):
        self.queue.put(QueueMessage(self.task_id, fitness, lio_fitness, lio_p), block=False)

    def __call__(self, mh, space, fn):
        self.queue.put(QueueMessage(self.task_id, space.best_agent.fit, None, None), block=False)

    def finish(self):
        self.queue.put(QueueMessage(self.task_id, None, None, None), block=False)