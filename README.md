# CS602 Algorithm Design and Implementation (G2 - Group 5)

### [Final presentation slides](https://docs.google.com/presentation/d/1wnqwfYSPhZJwy0msSXpv69tcGVAU7V-l1JVBjTRq1Kw/edit)


## Setup

```bash
conda env create -f environment.yml
conda activate algo-design
```


## Running the algorithm

Run `solve.py` on an instance file:

```bash
python solve.py submission_test_cases/grp5_small.txt
python solve.py submission_test_cases/grp5_small.txt --output grp5_small_sol.txt    # save output to file
```

### Calling the solver from Python

The algorithm is in `final_solution.py`. Call `run_algo`:

```python
run_algo(cust_demands, dist_matrix, vehicle_cap, num_vehicles, instance_type)
```

- `cust_demands`: list of demands, depot first (`[0, 2, 2, 3]`)
- `dist_matrix`: list of rows of the distance matrix
- `vehicle_cap`: capacity of each vehicle
- `num_vehicles`: number of vehicles available
- `instance_type`: `"small"`, `"medium"` or `"large"` (lowercase). The run stops at 95% of the time limit for that type (30 / 60 / 180 s).

It returns `(best_cost, best_routes)`, where `best_routes` is a list of routes.


## Input and output format

### Input:

Each instance is a plain text file with 5 parts, in this order:

1. Instance type: `small`, `medium` or `large` (any case)
2. Customer demands: on one line, separated by spaces. The first value is the depot (vertex `0`) and is always `0`.
3. Distance matrix: one row per line, values separated by spaces. It is symmetric, and row/column `0` is the depot.
4. Vehicle capacity, as `Capacity: <value>`
5. Number of vehicles, as `Vehicles: <value>`

Example:

```
small
0 2 2 3
0 2.5 5.6 6.0
2.5 0 3.9 6.5
5.6 3.9 0 5.2
6.0 6.5 5.2 0
Capacity: 5
Vehicles: 2
```


### Output: 

The total cost on the first line (to 6 decimal places), then one route per line, each starting and ending at the depot (`0`):

```
21.800000
0 1 0
0 3 2 0
```


## Submitted test cases

The 3 instances we are submitting are in `submission_test_cases/`:

| File | Type (time limit) | Customers | Vehicles | Capacity | Total demand | Tightness | Source |
|---|---|---|---|---|---|---|---|
| [`grp5_small.txt`](submission_test_cases/grp5_small.txt) | small (30 s) | 360 | 120 | 20 | 2,400 | 100.00% | made by our group (CW-trap-n361-k120) |
| [`grp5_medium.txt`](submission_test_cases/grp5_medium.txt) | medium (60 s) | 375 | 94 | 4 | 375 | 99.73% | X-n376-k94 (Uchoa et al., 2017) |
| [`grp5_large.txt`](submission_test_cases/grp5_large.txt) | large (180 s) | 654 | 131 | 5 | 654 | 99.85% | X-n655-k131 (Uchoa et al., 2017) |

Tightness = total demand ÷ (vehicles × capacity). Close to 100% means almost every vehicle must be full.

Our solver's results on them:

| File | Cost | Gap from optimal | Best found at |
|---|---|---|---|
| `grp5_small.txt` | 24,000,000 | 0.00% | 0.38 s |
| `grp5_medium.txt` | 147,766 | 0.04% | 45.54 s |
| `grp5_large.txt` | 107,035 | 0.24% | 61.30 s |


## Our approach

![Algorithm overview](images/algorithm_overview.png)


## Benchmarking

`run_benchmarks.py` runs `final_solution.py` on the instances in `test_cases/benchmarks` (sets A, B, P and X), checks that each solution is feasible, and compares its cost with the optimal cost in the matching `*_opt.txt` file.

```bash
python run_benchmarks.py                                  # every instance (about 3 min each)
python run_benchmarks.py --sets A B                       # only sets A and B
python run_benchmarks.py --sets X --limit 5               # the 5 smallest instances of set X
python run_benchmarks.py --instances A-n32-k5 X-n101-k25  # chosen instances
python run_benchmarks.py --limits 10 20 30                # quick check with shorter limits
```

Results are saved in `logs/`, and used to select the final test cases for submission.


## Team members
- Andrew Ho
- Amelia Lee
- Li Song Xi
- Oh Rui Han
- Peng You Yun