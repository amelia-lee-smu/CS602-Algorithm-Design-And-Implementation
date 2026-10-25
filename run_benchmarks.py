'''
Runs final_solution.run_algo on the benchmark instances, logs every checkpoint to one CSV,
and compares each result with the optimal solution given with the test case (*_opt.txt).

Usage (from the project folder):
    python run_benchmarks.py                                  # every instance, full time limits
    python run_benchmarks.py --sets A B                       # only sets A and B
    python run_benchmarks.py --sets X --limit 5               # the 5 smallest instances of set X
    python run_benchmarks.py --instances A-n32-k5 X-n101-k25  # chosen instances
    python run_benchmarks.py --time 10                        # stop each run after 10 s (quick check)

Output (in logs/):
    rfcs_log_<run_id>.csv     one row per checkpoint (columns: LOG_COLUMNS in utilities.py)
    summary_<run_id>.csv      one row per instance: cost, optimal cost, gap_pct, feasible, score
'''

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import final_solution
from utilities import calculate_cost, is_feasible, read_instance, read_solution

BENCHMARK_DIR = Path(__file__).parent / "test_cases" / "benchmarks"
LOG_DIR = Path(__file__).parent / "logs"

SUMMARY_COLUMNS = ["run_id", "instance", "instance_type", "num_customers", "num_vehicles",
                   "time_s", "cost", "optimal_cost", "gap_pct", "feasible", "score"]


def instance_type_for(num_customers):
    '''
    small / medium / large for an instance with this many customers.
    The spec does not say what counts as each size, so these thresholds are a GUESS.
    '''
    if num_customers < 180:
        return "small"
    if num_customers < 450:
        return "medium"
    return "large"


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark run_algo and log every checkpoint")
    parser.add_argument("--sets", nargs="+", default=["A", "B", "P", "X"])
    parser.add_argument("--instances", nargs="+", help="only these instances, e.g. A-n32-k5")
    parser.add_argument("--limit", type=int, help="at most this many instances per set (smallest first)")
    parser.add_argument("--time", type=float, help="seconds per run, instead of the full time limit")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = str(LOG_DIR / f"rfcs_log_{run_id}.csv")
    summary_path = LOG_DIR / f"summary_{run_id}.csv"
    LOG_DIR.mkdir(exist_ok=True)

    instances = []
    for set_name in args.sets:
        set_instances = find_instances([set_name], args.instances)
        instances += set_instances[:args.limit] if args.limit else set_instances

    with open(summary_path, "w", newline="") as summary_file:
        summary = csv.writer(summary_file)
        summary.writerow(SUMMARY_COLUMNS)

        print(f"{'instance':<16}{'type':<8}{'time':>7}{'cost':>11}{'optimal':>12}{'gap_pct':>9}  feasible  score")
        for path in instances:
            cust_demands, dist_matrix, vehicle_cap, num_vehicles = read_instance(path)
            num_customers = len(cust_demands) - 1
            instance_type = instance_type_for(num_customers)
            time_limit = final_solution.TIME_LIMITS[instance_type]

            # --time: shorten this run (run_algo stops at TIME_LIMIT_FRACTION x TIME_LIMITS[instance_type])
            if args.time:
                final_solution.TIME_LIMITS[instance_type] = args.time / final_solution.TIME_LIMIT_FRACTION

            log_info = {"log_path": log_path, "run_id": run_id,
                        "instance": path.stem, "instance_type": instance_type}
            start = time.monotonic()
            cost, routes = final_solution.run_algo(cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                                                   instance_type, log_info)
            run_time = time.monotonic() - start
            final_solution.TIME_LIMITS[instance_type] = time_limit     # put the real limit back

            opt_path = path.with_name(path.stem + "_opt.txt")
            optimal_cost = read_solution(opt_path)[0] if opt_path.exists() else None

            feasible = (routes is not None and is_feasible(routes, cust_demands, vehicle_cap, num_vehicles)
                        and abs(calculate_cost(dist_matrix, routes) - cost) < 1e-6)
            # gap_pct: how far this solution is above the optimal cost, in %
            gap_pct = 100 * (cost - optimal_cost) / optimal_cost if feasible and optimal_cost else None
            score = project_score(run_time, time_limit, cost, optimal_cost) if feasible and optimal_cost else 0

            summary.writerow([run_id, path.stem, instance_type, num_customers, num_vehicles,
                              f"{run_time:.2f}", f"{cost:.2f}" if feasible else "",
                              optimal_cost if optimal_cost is not None else "",
                              f"{gap_pct:.2f}" if gap_pct is not None else "", "yes" if feasible else "no",
                              f"{score:.3f}"])
            summary_file.flush()

            cost_text = f"{cost:.0f}" if feasible else "none"
            gap_text = f"{gap_pct:.1f}%" if gap_pct is not None else "-"
            print(f"{path.stem:<16}{instance_type:<8}{run_time:6.1f}s{cost_text:>11}{str(optimal_cost):>12}"
                  f"{gap_text:>9}  {'yes' if feasible else 'no':<8}  {score:.3f}")

    print(f"\ncheckpoint log: {os.path.relpath(log_path)}")
    print(f"summary:        {os.path.relpath(summary_path)}")


if __name__ == "__main__":
    main()
