'''
Clarke-Wright savings + penalised local search for CVRP.

Pipeline (all steps are written out in run_cw_ls_algo below; the moves
themselves are in optimisations.py):
  1. START      Build routes with the Clarke-Wright savings algorithm.
  2. IMPROVE    Penalised local search:
                  (a) reduce_routes: cut the routes down to the number of vehicles
                      (customers are re-inserted, vehicles may become overloaded)
                  (b) local_search (relocate / swap / 2-opt* / 2-opt) on
                      "distance + penalty * overload"
                  (c) no vehicle overloaded? done.
                      otherwise double the penalty, fix_overload, back to (b)
  3. RESTART    Still infeasible and time left? Restart from a randomised
                Clarke-Wright (each saving multiplied by a factor in [0.8, 1.2]).
  4. FINISH     Check feasibility with utilities.is_feasible and print.

Every step checks the clock and stops before the time budget.
'''
import argparse
import contextlib
import random
import sys
import time

from utilities import read_instance, is_feasible, calculate_cost, size_class, TIME_LIMITS
from optimisations import (new_search, set_routes, snapshot, reduce_routes, local_search,
                           fix_overload, overload, distance, num_routes, out_of_time, log)

SEED = 42
SAFETY = 0.8          # use 80% of the time limit, leaving room for reading input and printing
MAX_ROUNDS = 30       # max rounds of "local search, then raise the penalty"


# ------------------------------------------------------------------ construction
def clarke_wright(cust_demands, dist_matrix, vehicle_capacity, rng=None, noise=0.0, lam=1.0):
    '''
    Clarke-Wright savings algorithm. Returns a list of routes, each a list of
    customers WITHOUT the depot. Ignores the vehicle limit (fixed later).

    rng, noise: optional. If given, every saving is multiplied by a random
    factor in [1 - noise, 1 + noise], giving a slightly different solution
    each call (used for restarts).
    lam:        saving = d(0,i) + d(0,j) - lam * d(i,j). lam = 1 is the classic
                saving; a larger lam favours joining customers that are close
                to each other, a smaller one favours customers far from the depot.
    '''
    n = len(cust_demands)

    # (1) every customer gets its own route: 0 -> c -> 0
    routes = {c: [c] for c in range(1, n)}              # route id -> customers
    load = {c: cust_demands[c] for c in range(1, n)}    # route id -> total demand
    route_of = {c: c for c in range(1, n)}              # customer -> route id

    # (2) saving of serving i and j on the same route: d(0,i) + d(0,j) - lam * d(i,j)
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = dist_matrix[0][i] + dist_matrix[0][j] - lam * dist_matrix[i][j]
            if s > 0:
                if rng is not None:
                    s *= rng.uniform(1 - noise, 1 + noise)
                savings.append((s, i, j))

    # (3) biggest savings first
    savings.sort(reverse=True)

    # (4) merge the route of i with the route of j where allowed
    for _, i, j in savings:
        a, b = route_of[i], route_of[j]
        if a == b:                                          # already on the same route
            continue
        if load[a] + load[b] > vehicle_capacity:            # merged route must fit
            continue
        route_a, route_b = routes[a], routes[b]
        # i and j must each be at an END of their route (next to the depot)
        if i not in (route_a[0], route_a[-1]) or j not in (route_b[0], route_b[-1]):
            continue
        # orient so that route_a ends with i and route_b starts with j
        if route_a[-1] != i:
            route_a.reverse()
        if route_b[0] != j:
            route_b.reverse()
        route_a.extend(route_b)
        load[a] += load[b]
        for c in route_b:
            route_of[c] = a
        del routes[b], load[b]

    return list(routes.values())


# ------------------------------------------------------------------ helpers
def with_depot(routes):
    '''[3, 1] -> [0, 3, 1, 0], the format used by utilities.py and the output.'''
    return [[0] + r + [0] for r in routes]


def time_budget(num_customers):
    '''Seconds we allow ourselves for an instance of this size.'''
    return SAFETY * TIME_LIMITS[size_class(num_customers)]


# ------------------------------------------------------------------ solver
def run_cw_ls_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
                   time_limit=None, seed=SEED, verbose=False):
    '''
    Clarke-Wright + penalised local search, with randomised restarts if the
    result is infeasible. Returns (cost, routes) with routes including the depot.
    '''
    start = time.perf_counter()
    if time_limit is None:
        time_limit = time_budget(len(cust_demands) - 1)
    deadline = start + time_limit

    search = new_search(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
                        deadline, seed=seed, verbose=verbose)
    rng = random.Random(seed)

    def feasible(routes):
        # quick check without printing (local search never drops a customer,
        # so only the vehicle count and capacities can be violated)
        return (len(routes) <= num_vehicles and
                all(sum(cust_demands[c] for c in r) <= vehicle_capacity for r in routes))

    restarts = 0
    while True:
        # (1) START: plain Clarke-Wright first, randomised Clarke-Wright on restarts
        if restarts == 0:
            start_routes = clarke_wright(cust_demands, dist_matrix, vehicle_capacity)
        else:
            start_routes = clarke_wright(cust_demands, dist_matrix, vehicle_capacity,
                                         rng=rng, noise=0.2)
        search['penalty'] = search['initial_penalty']
        set_routes(search, start_routes)
        best, best_cost = None, float('inf')        # best feasible routes of this attempt
        if feasible(start_routes):
            best, best_cost = snapshot(search), distance(search)
        log(search, f'start: {num_routes(search)} routes, cost {distance(search):.1f}')

        # (2a) too many routes: delete the lightest ones, re-insert their customers
        reduce_routes(search)
        log(search, f'after reduce_routes: {num_routes(search)} routes, '
                    f'overload {overload(search)}, cost {distance(search):.1f}')

        for rnd in range(MAX_ROUNDS):
            # (2b) local search: relocate / swap / 2-opt* / 2-opt until no move helps
            local_search(search)
            log(search, f'round {rnd}: penalty {search["penalty"]:.3g}, '
                        f'cost {distance(search):.1f}, overload {overload(search)}')

            # (2c) feasible: done. overloaded: raise the penalty and repair, then search again
            if overload(search) == 0:
                if distance(search) < best_cost:
                    best, best_cost = snapshot(search), distance(search)
                break
            if out_of_time(search):
                break
            search['penalty'] *= 2
            fix_overload(search)

        routes = best if best is not None else snapshot(search)

        # (3) RESTART only if still infeasible and there is time left
        if feasible(routes) or out_of_time(search):
            break
        restarts += 1
        log(search, f'restart {restarts}')

    routes = with_depot(routes)
    log(search, f'restarts: {restarts}, time {time.perf_counter() - start:.2f}s')
    return calculate_cost(routes, dist_matrix), routes


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description='Clarke-Wright + local search CVRP solver')
    parser.add_argument('instance', nargs='?', help='instance file (default: sample from instructions)')
    parser.add_argument('--time', type=float, help='time budget in seconds (default: from instance size)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--verbose', action='store_true', help='print progress')
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.instance:
        cust_demands, dist_matrix, vehicle_capacity, num_vehicles = read_instance(args.instance)
    else:
        from test_cases.sample_from_instructions import (
            cust_demands, dist_matrix, vehicle_capacity, num_vehicles)

    # the time budget counts from program start, so reading the file is included
    time_limit = args.time if args.time else time_budget(len(cust_demands) - 1)
    time_limit -= time.perf_counter() - t0

    cost, routes = run_cw_ls_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
                                  time_limit=time_limit, seed=args.seed, verbose=args.verbose)

    # is_feasible prints its reason; send that to stderr so stdout stays in the output format
    with contextlib.redirect_stdout(sys.stderr):
        if not is_feasible(routes, cust_demands, vehicle_capacity, num_vehicles):
            print('WARNING: no feasible solution found')

    print(round(cost, 4))
    for route in routes:
        print(' '.join(str(v) for v in route))


if __name__ == '__main__':
    main()
