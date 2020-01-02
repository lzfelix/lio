import csv
import glob
import pickle
import argparse
from pathlib import Path
from tqdm import auto as tqdm
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def read_all_pickles(path: str) -> tuple:
    """Reads all files within a given folder, expecting them to be pickles."""
    result_pickles = glob.glob(str(path) + '/*')

    all_qpso = []
    all_lio = []
    all_lio_time = []
    all_qpso_time = []
    all_lio_p = []

    for file in result_pickles:
        data = pickle.load(Path(file).open('rb'))

        qpso_fitness = data['opt_history'].best_agent[-1][-1]
        qpso_time = data['opt_history'].time

        lio_fitness = data['lio_history'].best_agent[-1][-1]
        lio_p = np.asarray(data['lio_history'].best_agent[-1][0]).ravel().item()
        lio_time = data['lio_history'].time

        all_qpso.append(qpso_fitness)
        all_qpso_time.append(qpso_time)

        all_lio.append(lio_fitness)
        all_lio_p.append(lio_p)
        all_lio_time.append(lio_time)
    return all_qpso, all_qpso_time, all_lio, all_lio_p, all_lio_time


def parse_all_results(results_folder: str) -> dict:
    """Given a folder with the following structure:

    ```
        reults/
            sphere/
                10/
                    experiment_1.pickle
                    ...
                    experiment_n.pickle
                25/
            ackley1/
                ...
    ```
    parses each pickle file and computes several statistics
    """
    stats = defaultdict(lambda: defaultdict(dict))
    base_path = Path(results_folder)

    fn_iterator = tqdm.tqdm(list(base_path.iterdir()))
    fn_iterator.set_description('Parsing functions')
    for fn_dir in fn_iterator:
        fn_name = fn_dir.stem
        if fn_name[0] == '.':
            continue
        fn_iterator.set_postfix(dict(fun=fn_name))

        # Sorting dims as 10, 25, 100, ...
        folders = list(fn_dir.iterdir())
        folders = sorted(folders, key=lambda s: int(s.stem))
        for fn_at_dim_dir in folders:
            dim_name = int(fn_at_dim_dir.stem)
            all_qpso, qpso_time, all_lio, all_lio_p, lio_time \
                = read_all_pickles(fn_at_dim_dir)

            stats[fn_name][dim_name] = {
                'qpso': np.mean(all_qpso),
                'qpso_std': np.std(all_qpso),
                'qpso_time': np.mean(qpso_time),
                'qpso_time_std': np.std(qpso_time),
                'lio': np.mean(all_lio),
                'lio_std': np.std(all_lio),
                'lio_time': np.mean(lio_time),
                'lio_time_std': np.std(lio_time),
                'p': np.mean(all_lio_p),
                'p_std': np.std(all_lio_p),
                'wilcoxon': wilcoxon(all_qpso, all_lio)[1],
            }
    return stats


def results_to_csv(destination: str, stats: dict) -> None:
    """Stores all experimental statistics into a single csv file."""
    writing_pointer = open(destination, 'w')
    writer = csv.DictWriter(
        writing_pointer,
        ['fn', 'dims', 'qpso', 'qpso_std', 'qpso_time', 'qpso_time_std',
         'lio', 'lio_std', 'lio_time', 'lio_time_std', 'p', 'p_std',
         'wilcoxon']
    )
    writer.writeheader()

    for fn_name, experiments in stats.items():
        for dim, experiment in experiments.items():
            writer.writerow({
                'fn': fn_name,
                'dims': dim,
                'qpso': experiment['qpso'],
                'qpso_std': experiment['qpso_std'],
                'qpso_time': experiment['qpso_time'],
                'qpso_time_std': experiment['qpso_time_std'],
                'lio': experiment['lio'],
                'lio_std': experiment['lio_std'],
                'lio_time': experiment['lio_time'],
                'lio_time_std': experiment['lio_time_std'],
                'p': experiment['p'],
                'p_std': experiment['p_std'],
                'wilcoxon': experiment['wilcoxon'],
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compiles all experimental' +
                                     'results from run_experiment.py' +
                                     'into a single csv file')
    parser.add_argument('results_dir', help='Folder containing all experimental results. ' +
                        'Ex: ./results/')
    parser.add_argument('dest_csv', help='Full path to the destination CSV file.')
    exec_args = parser.parse_args()

    all_results = parse_all_results(exec_args.results_dir)
    results_to_csv(exec_args.dest_csv, all_results)
