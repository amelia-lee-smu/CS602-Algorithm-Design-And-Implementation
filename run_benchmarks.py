'''
Runs the solver on the benchmark instances and compares it with the optimal solution given with
each test case (*_opt.txt), at every time limit in the spec.

Each instance runs ONCE, for the longest time limit (180 s). The best solution so far is read at
TIME_LIMIT_FRACTION x each limit (28.5 s, 57 s, 171 s), the same moments run_algo would stop
a small, medium or large run. So one run shows how far from optimal the solver is at each limit,
and no instance size has to be guessed.

Usage (from the project folder):
    python run_benchmarks.py                                  # every instance (about 3 min each)
    python run_benchmarks.py --sets A B                       # only sets A and B
    python run_benchmarks.py --sets X --limit 5               # the 5 smallest instances of set X
    python run_benchmarks.py --instances A-n32-k5 X-n101-k25  # chosen instances
    python run_benchmarks.py --limits 10 20 30                  # quick check with shorter limits

Output (in logs/, written by test_case_logger.py):
    rfcs_log_<run_id>.csv     one row per solver step (columns: LOG_COLUMNS in test_case_logger.py)
    summary_<run_id>.csv      one row per instance and time limit (columns: SUMMARY_COLUMNS in test_case_logger.py)

Instances that can never be solved (see instance_infeasibility) are skipped: the solver is not run,
and the log gets one "instance_infeasible" row saying why.
'''

import argparse
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

import final_solution
import test_case_logger

BENCHMARK_DIR = Path(__file__).parent / "test_cases" / "benchmarks"
LOG_DIR = Path(__file__).parent / "logs"


# ---------------------------------------------------------------------------
# Reading instances and solutions
# ---------------------------------------------------------------------------

def to_number(text):
    '''Converts "5" to 5 and "2.5" to 2.5'''
    value = float(text)
    return int(value) if value.is_integer() else value


def read_instance(path):
    '''
    Reads an instance file in the project format: a line of demands (depot first),
    then the distance matrix, then "Capacity: ..." and "Vehicles: ...".
    Returns (cust_demands, dist_matrix, vehicle_cap, num_vehicles)
    '''
    with open(path) as instance_file:
        lines = [line.strip() for line in instance_file if line.strip()]

    vehicle_cap = num_vehicles = None
    number_rows = []
    for line in lines:
        label = line.lower()
        if label.startswith("capacity"):
            vehicle_cap = to_number(line.split(":")[1])
        elif label.startswith("vehicles"):
            num_vehicles = int(line.split(":")[1])
        else:
            number_rows.append([to_number(value) for value in line.split()])

    cust_demands, dist_matrix = number_rows[0], number_rows[1:]
    return cust_demands, dist_matrix, vehicle_cap, num_vehicles


def read_solution(path):
    '''
    Reads a solution file: the cost on the first line, then one route per line.
    Returns (cost, routes)
    '''
    with open(path) as solution_file:
        lines = [line.strip() for line in solution_file if line.strip()]
    cost = to_number(lines[0])
    routes = [[int(stop) for stop in line.split()] for line in lines[1:]]
    return cost, routes


def instance_infeasibility(cust_demands, vehicle_capacity, num_vehicles):
    '''
    Checks whether the instance can ever be solved, whatever the routes:
    1. no single customer may need more than vehicle_capacity
    2. the total demand must fit in num_vehicles x vehicle_capacity
    Only used for testing: the solver can assume every instance it is given is feasible.
    Returns "" if it may be solvable, otherwise why it cannot be, e.g. "customer 12 demand 120 > capacity 100"
    '''
    heaviest = max(cust_demands)
    if heaviest > vehicle_capacity:
        return f"customer {cust_demands.index(heaviest)} demand {heaviest} > capacity {vehicle_capacity}"

    total_demand = sum(cust_demands)
    if total_demand > num_vehicles * vehicle_capacity:
        return f"total demand {total_demand} > {num_vehicles} vehicles x capacity {vehicle_capacity}"

    return ""


# ---------------------------------------------------------------------------
# Running the benchmarks
# ---------------------------------------------------------------------------


def num_customers_in_name(path):
    '''e.g. A-n32-k5 -> 31 customers (n counts the depot); used to sort instances by size'''
    try:
        return int(path.stem.split("-n")[1].split("-")[0]) - 1
    except (IndexError, ValueError):
        return 0


def find_instances(sets, names):
    '''Instance files (not the *_opt.txt solutions), smallest first within each set.'''
    instances = []
    for set_name in sets:
        files = [path for path in (BENCHMARK_DIR / set_name).glob("*.txt") if not path.stem.endswith("_opt")]
        instances += sorted(files, key=num_customers_in_name)
    if names:
        instances = [path for path in instances if path.stem in names]
    return instances


def run_with_snapshots(cust_demands, dist_matrix, vehicle_cap, num_vehicles, time_limits, log_info):
    '''
    Runs final_solution.find_solution once (through test_case_logger, which logs each step),
    the same way run_algo does (separate process, shared best),
    and reads the best solution at TIME_LIMIT_FRACTION x each time limit.
    The process is terminated after the last one.
    Returns [(time_limit, run_time, cost, routes, found_at)], one per time limit:
        run_time = seconds into the run when the best solution was read
        found_at = seconds into the run when that solution was first found (None if not known)
    '''
    snapshots = []
    with multiprocessing.Manager() as manager:
        best = manager.dict(solution=(float("inf"), None))
        start = time.monotonic()
        log_info = {**log_info, "run_start": start}     # so test_case_logger times each new best from the same start
        process = multiprocessing.Process(
            target=test_case_logger.find_solution_with_log,
            args=(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info),
        )
        process.start()

        for time_limit in time_limits:
            stop_at = start + final_solution.TIME_LIMIT_FRACTION * time_limit
            process.join(timeout=max(0, stop_at - time.monotonic()))   # returns early if the solver ends
            run_time = time.monotonic() - start

            # read the solution and when it was found together; if a new best lands between the two
            # (the time is written just after the solution), read again
            for _ in range(10):
                current = best.copy()
                cost, routes = current["solution"]
                found_cost, found_at = current.get("found", (None, None))
                if found_cost == cost:
                    break
                time.sleep(0.001)
            else:
                found_at = None
            snapshots.append((time_limit, run_time, cost, routes, found_at))

        if process.is_alive():
            process.terminate()
            process.join()
    return snapshots


def main():
    parser = argparse.ArgumentParser(description="Benchmark the solver at each time limit")
    parser.add_argument("--sets", nargs="+", default=["A", "B", "P", "X"])
    parser.add_argument("--instances", nargs="+", help="only these instances, e.g. A-n32-k5")
    parser.add_argument("--limit", type=int, help="at most this many instances per set (smallest first)")
    parser.add_argument("--limits", nargs="+", type=float,
                        default=sorted(final_solution.TIME_LIMITS.values()),
                        help="time limits in seconds to report (default: the spec's 30 60 180)")
    args = parser.parse_args()
    time_limits = sorted(args.limits)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = str(LOG_DIR / f"rfcs_log_{run_id}.csv")
    summary_path = LOG_DIR / f"summary_{run_id}.csv"
    LOG_DIR.mkdir(exist_ok=True)

    instances = []
    for set_name in args.sets:
        set_instances = find_instances([set_name], args.instances)
        instances += set_instances[:args.limit] if args.limit else set_instances

    minutes = len(instances) * final_solution.TIME_LIMIT_FRACTION * time_limits[-1] / 60
    print(f"{len(instances)} instances, about {minutes:.0f} min in total\n")

    summary = test_case_logger.open_summary(str(summary_path))

    # console: one line per instance, the gap from optimal at each time limit
    gap_headers = "".join(f"{f'gap@{limit:g}s':>11}" for limit in time_limits)
    print(f"{'instance':<16}{'optimal':>10}{gap_headers}{'cost':>11}")

    for path in instances:
        cust_demands, dist_matrix, vehicle_cap, num_vehicles = read_instance(path)
        num_customers = len(cust_demands) - 1

        opt_path = path.with_name(path.stem + "_opt.txt")
        optimal_cost = read_solution(opt_path)[0] if opt_path.exists() else None

        log_info = {"log_path": log_path, "instance": path.stem}

        # skip instances that can never be solved: log why, and do not run the solver
        impossible = instance_infeasibility(cust_demands, vehicle_cap, num_vehicles)
        if impossible:
            test_case_logger.write_skipped_instance(summary, log_info, run_id, path.stem, num_customers, num_vehicles,
                                              optimal_cost, time_limits, impossible)
            print(f"{path.stem:<16}{str(optimal_cost):>10}  skipped, instance infeasible: {impossible}")
            continue

        snapshots = run_with_snapshots(cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                                       time_limits, log_info)
        gap_texts = test_case_logger.write_summary_rows(summary, run_id, path.stem, cust_demands, dist_matrix,
                                                  vehicle_cap, num_vehicles, optimal_cost, snapshots)

        final_cost = snapshots[-1][2]
        cost_text = f"{final_cost:.0f}" if final_cost != float("inf") else "none"
        print(f"{path.stem:<16}{str(optimal_cost):>10}{''.join(f'{text:>11}' for text in gap_texts)}"
              f"{cost_text:>11}")

    summary["file"].close()

    print(f"\ncheckpoint log: {os.path.relpath(log_path)}")
    print(f"summary:        {os.path.relpath(summary_path)}")


if __name__ == "__main__":
    main()
