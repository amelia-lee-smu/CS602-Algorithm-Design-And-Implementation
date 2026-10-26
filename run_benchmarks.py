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
    python run_benchmarks.py --limits 3 6 18                  # quick check with shorter limits

Output (in logs/):
    rfcs_log_<run_id>.csv     one row per solver step (columns: LOG_COLUMNS in solver_log.py)
    summary_<run_id>.csv      one row per instance and time limit: cost, gap_pct, feasible, score

Instances that can never be solved (see instance_infeasibility) are skipped: the solver is not run,
and the log gets one "instance_infeasible" row saying why.
'''

import argparse
import csv
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

import final_solution
import solver_log
from utilities import calculate_cost, is_feasible

BENCHMARK_DIR = Path(__file__).parent / "test_cases" / "benchmarks"
LOG_DIR = Path(__file__).parent / "logs"

SUMMARY_COLUMNS = ["run_id", "instance", "num_customers", "num_vehicles", "time_limit_s", "elapsed_s",
                   "cost", "optimal_cost", "gap_pct", "feasible", "score"]


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


def project_score(run_time, time_limit, cost, optimal_cost):
    '''Score from the project spec, using the optimal cost in place of the best team's cost.'''
    if run_time > 1.5 * time_limit:
        return 0.25
    time_term = 2 ** max(0, run_time / time_limit - 1)
    return 1 / (0.5 * (time_term + cost / optimal_cost))


def run_with_snapshots(cust_demands, dist_matrix, vehicle_cap, num_vehicles, time_limits, log_info):
    '''
    Runs final_solution.find_solution once (through solver_log, which logs each step),
    the same way run_algo does (separate process, shared best),
    and reads the best solution at TIME_LIMIT_FRACTION x each time limit.
    The process is terminated after the last one.
    Returns [(time_limit, elapsed seconds, cost, routes)], one per time limit
    '''
    snapshots = []
    with multiprocessing.Manager() as manager:
        best = manager.dict(solution=(float("inf"), None))
        process = multiprocessing.Process(
            target=solver_log.find_solution_with_log,
            args=(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info),
        )
        start = time.monotonic()
        process.start()

        for time_limit in time_limits:
            stop_at = start + final_solution.TIME_LIMIT_FRACTION * time_limit
            process.join(timeout=max(0, stop_at - time.monotonic()))   # returns early if the solver ends
            cost, routes = best["solution"]
            snapshots.append((time_limit, time.monotonic() - start, cost, routes))

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

    with open(summary_path, "w", newline="") as summary_file:
        summary = csv.writer(summary_file)
        summary.writerow(SUMMARY_COLUMNS)

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
                log = solver_log.open_log(log_info)
                solver_log.write_row(log, 0, "instance_infeasible", None, float("inf"), False,
                                     None, num_vehicles, impossible)
                log["file"].close()
                for time_limit in time_limits:
                    summary.writerow([run_id, path.stem, num_customers, num_vehicles, f"{time_limit:g}",
                                      "0.00", "", optimal_cost if optimal_cost is not None else "",
                                      "", "no", "0.000"])
                summary_file.flush()
                print(f"{path.stem:<16}{str(optimal_cost):>10}  skipped, instance infeasible: {impossible}")
                continue

            snapshots = run_with_snapshots(cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                                           time_limits, log_info)

            gap_texts = []
            for time_limit, elapsed, cost, routes in snapshots:
                feasible = (routes is not None and is_feasible(routes, cust_demands, vehicle_cap, num_vehicles)
                            and abs(calculate_cost(dist_matrix, routes) - cost) < 1e-6)
                # gap_pct: how far this solution is above the optimal cost, in %
                gap_pct = 100 * (cost - optimal_cost) / optimal_cost if feasible and optimal_cost else None
                score = project_score(elapsed, time_limit, cost, optimal_cost) if feasible and optimal_cost else 0

                summary.writerow([run_id, path.stem, num_customers, num_vehicles, f"{time_limit:g}",
                                  f"{elapsed:.2f}", f"{cost:.2f}" if feasible else "",
                                  optimal_cost if optimal_cost is not None else "",
                                  f"{gap_pct:.2f}" if gap_pct is not None else "",
                                  "yes" if feasible else "no", f"{score:.3f}"])
                if gap_pct is not None:
                    gap_texts.append(f"{gap_pct:.2f}%")
                else:
                    gap_texts.append("infeasible" if not feasible else "-")
            summary_file.flush()

            final_cost = snapshots[-1][2]
            cost_text = f"{final_cost:.0f}" if final_cost != float("inf") else "none"
            print(f"{path.stem:<16}{str(optimal_cost):>10}{''.join(f'{text:>11}' for text in gap_texts)}"
                  f"{cost_text:>11}")

    print(f"\ncheckpoint log: {os.path.relpath(log_path)}")
    print(f"summary:        {os.path.relpath(summary_path)}")


if __name__ == "__main__":
    main()
