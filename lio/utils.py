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
