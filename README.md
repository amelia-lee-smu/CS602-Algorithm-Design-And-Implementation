# CS602 Algorithm Design and Implementation (G2 - Group 5)

## Setup

```bash
conda env create -f environment.yml
conda activate algo-design
```


## Input and output format

### Output: 

The total cost on the first line, then one route per line, each starting and ending at the depot (`0`):

```
21.8
0 1 0
0 2 3 0
```


## Algorithms

1. `route_first_cluster_second_solution.py` implements the **route first, cluster second** method from Beasley (1983): 
   - Build a random giant tour through all customers (excluding the depot) and improve it with 2-opt.
   - Partition the giant tour
   - optimise each route using 3-opt
   - (Note: multiple giant tours are created, to reduce the effect of having a bad random start)

2. `clarke_wright_solution.py` implements **Clarke-Wright savings + penalised local search**:
   - Build routes with the Clarke-Wright savings algorithm.
   - Cut the routes down to the number of vehicles (re-inserting customers; vehicles may become overloaded).
   - Local search (relocate, swap, 2-opt*, 2-opt over each customer's 30 nearest neighbours) on `distance + penalty * overload`, doubling the penalty until no vehicle is overloaded (`PenalisedLocalSearch` in `optimisations.py`).
   - If still infeasible, restart from a randomised Clarke-Wright.

   ```bash
   python clarke_wright_solution.py                                   # sample from the instructions
   python clarke_wright_solution.py test_cases/benchmarks/A/A-n32-k5.txt [--time 10] [--seed 42] [--verbose]
   ```

## Benchmarking

`run_benchmarks.py` runs a solver on `test_cases/benchmarks`, checks feasibility and scores it against the `*_opt.txt` best-known solutions with `utilities.score`:

```bash
python run_benchmarks.py --sets A B --time 5        # quick check, 5 s cap per instance
python run_benchmarks.py --sets X --csv results.csv  # full time budget, save results
```

To add another algorithm, expose `solve(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, time_limit) -> (cost, routes)` and register it in `SOLVERS`.

Size classes (small / medium / large) are set by `utilities.size_class`; the thresholds are a guess until the course defines them.
