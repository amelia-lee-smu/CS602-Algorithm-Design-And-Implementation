'''
Runs a solver on the benchmark instances and scores it against the best known
solutions (*_opt.txt), using the scoring rule from utilities.score().

Usage:
    python run_benchmarks.py                          # every instance in every set
    python run_benchmarks.py --sets A B               # only sets A and B
    python run_benchmarks.py --sets X --limit 5       # first 5 instances of set X
    python run_benchmarks.py --sets A --time 5        # cap each run at 5 s (quick check)
    python run_benchmarks.py --csv results.csv        # also save the table

Adding another solver: write a function with the signature
    solve(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, time_limit) -> (cost, routes)
(routes including the depot) and add it to SOLVERS below.
'''
import argparse
import contextlib
import csv
import io
import time
from pathlib import Path

from utilities import read_instance, read_solution, is_feasible, calculate_cost, score, size_class
from clarke_wright_solution import run_cw_ls_algo, time_budget

BENCHMARK_DIR = Path(__file__).parent / 'test_cases' / 'benchmarks'

SOLVERS = {
    'cw_ls': lambda dem, dist, cap, veh, t: run_cw_ls_algo(dem, dist, cap, veh, time_limit=t),
}


def find_instances(sets):
    '''Yields (instance path, opt solution path or None), sorted by set then size.'''
    def size(p):
        try:
            return int(p.stem.split('-n')[1].split('-')[0])
        except (IndexError, ValueError):
            return 0
    for s in sets:
        files = [p for p in (BENCHMARK_DIR / s).glob('*.txt') if not p.stem.endswith('_opt')]
        for p in sorted(files, key=size):
            opt = p.with_name(p.stem + '_opt.txt')
            yield p, (opt if opt.exists() else None)


def main():
    parser = argparse.ArgumentParser(description='Benchmark a CVRP solver')
    parser.add_argument('--solver', default='cw_ls', choices=SOLVERS)
    parser.add_argument('--sets', nargs='+', default=['A', 'B', 'P', 'X'])
    parser.add_argument('--limit', type=int, help='max instances per set')
    parser.add_argument('--time', type=float, help='time cap per instance in seconds '
                        '(default: the solver budget for the instance size)')
    parser.add_argument('--csv', help='save results to this CSV file')
    args = parser.parse_args()
    solve = SOLVERS[args.solver]

    header = f"{'instance':<16}{'size':<8}{'cost':>10}{'best':>10}{'gap%':>8}{'time':>8}  {'feasible':<9}{'score':>6}"
    print(header)
    print('-' * len(header))

    rows = []
    for s in args.sets:
        instances = list(find_instances([s]))
        if args.limit:
            instances = instances[:args.limit]
        for path, opt_path in instances:
            dem, dist, cap, veh = read_instance(path)
            n = len(dem) - 1
            sml = size_class(n)
            budget = min(args.time, time_budget(n)) if args.time else time_budget(n)

            t0 = time.perf_counter()
            cost, routes = solve(dem, dist, cap, veh, budget)
            elapsed = time.perf_counter() - t0

            with contextlib.redirect_stdout(io.StringIO()):     # hide is_feasible's message
                feasible = is_feasible(routes, dem, cap, veh)
            cost = calculate_cost(routes, dist)                 # recompute independently

            best = gap = final = None
            if opt_path:
                best, _ = read_solution(opt_path)
                gap = 100 * (cost - best) / best
                final = score(best, cost, sml, elapsed)[3] if feasible else 0.0

            fmt = lambda x, spec: format(x, spec) if x is not None else '-'
            print(f"{path.stem:<16}{sml:<8}{cost:>10.1f}{fmt(best, '>10.1f')}{fmt(gap, '>8.2f')}"
                  f"{elapsed:>8.1f}  {str(feasible):<9}{fmt(final, '>6.3f')}", flush=True)
            rows.append({'set': s, 'instance': path.stem, 'customers': n, 'size': sml,
                         'cost': round(cost, 4), 'best': best, 'gap_pct': gap,
                         'time_s': round(elapsed, 2), 'feasible': feasible, 'score': final})

    # summary: average score per size class, and the weighted final (0.1 / 0.3 / 0.6)
    print()
    weights = {'small': 0.1, 'medium': 0.3, 'large': 0.6}
    avgs = {}
    for sml in weights:
        scores = [r['score'] for r in rows if r['size'] == sml and r['score'] is not None]
        if scores:
            avgs[sml] = sum(scores) / len(scores)
            print(f'{sml:<7} avg score {avgs[sml]:.4f}  over {len(scores)} instances')
    infeasible = [r['instance'] for r in rows if not r['feasible']]
    print(f'infeasible: {len(infeasible)}' + (f' ({", ".join(infeasible)})' if infeasible else ''))
    if avgs:
        w = sum(weights[k] for k in avgs)
        print(f'weighted score (over the classes present): {sum(weights[k] * v for k, v in avgs.items()) / w:.4f}')

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'saved {args.csv}')


if __name__ == '__main__':
    main()
