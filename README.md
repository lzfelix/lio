# Enhancing Hyper-To-Real Space Projections Through Euclidean Norm Meta-Heuristic Optimization

_Official paper implementation for Last Iteration Optimization (LIO) hypercomplex optimization
fine-tuning_

## Folders structure
  - `notebooks/`: Contains a single notebook briefly illustrating the idea proposed in the paper.
  - `lio/`: Contains the code related to the proposed approach (in `optimization.py`), as well as
    the script to reproduce the provided results (`run_all.sh`) and compile them in a single csv
    file (`compile_results.py`). For further instructions, please read below.

## Requirements
  - This code requires Python >= 3.6;
  - Please run `pip install -r requirements.txt` to fulfill the requirements.

## Adopting LIO to your pipeline

The overall idea of function optimization and `p`-value optimization are detailed in the script
`optimization.py`. Meanwhile, the function `run_experiment(...)` in the script `run_experiment.py`
shows how to integrate global and LIO optimizations in a single, straightforward, function for
optimization. Remaining functions in this script concern efficient parallel experiments execution.

## Reproducing the paper results

There are two options to reproduce the reported results. The first consists in optimizing a single
benchmarking function for a single amount of variables. To such an extent use the script
`run_experiment.py` by supplying the provided arguments that are shown by running `run_experiment.py --help`.
To compute the optimization statistics for 15 runs of the Brown function using 25 dimensions, for
instance, use the following command:

```bash
python run_experiment brown -n_agents 10 -n_vars 25 -n_iters 1000 -n_runs 15
```

To run all experiments as reported in the paper, just execute the shell script `run_all.sh`, which
will perform all experiments and store the results in `lio/results/`. This process may take a while,
but it is possible to observe its progress via command line. Further, use
`python compile_results.py ./results/ compiled_csv_file.csv` to compile all results in a single file.

## Optimizing new functions

Brand new functions can be optimized by using the proposed approach by editting the file `lio/benchmark.py`.
Just add a new function following the signature `def target_fn(numpy.ndarrray) -> float` by following the
already implemented functions. Further, register it in the `get_function(...)` function at the bottom of
this file, along with its lower and upper bounds. Currently all decision variables share the same limit values.

After following these steps, the function can be optimized via command line through the script
`run_experiment.py` (please refer to the "Reproducing paper results" section for more infomration
about it).

## Citation

If you use LIO (either this implementation or its overrall idea), please cite us as follows:

```bibtex
TBD
```
