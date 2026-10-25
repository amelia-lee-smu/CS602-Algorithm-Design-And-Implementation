'''
Clarke-Wright + ruin-and-recreate for CVRP (improved clarke_wright_solution.py).

Idea in one line: the score does not reward finishing early (any run-time up
to the time limit scores the same), so instead of stopping at the first
feasible solution we keep improving it until the time budget is used up.

Pipeline (all steps are written out in run_yy_cw_algo below; the moves
themselves are in optimisations.py):
  1. START      Several Clarke-Wright solutions, each with slightly different
                savings. Each one gets the penalised local search:
                  (a) reduce_routes: cut the routes down to the number of vehicles
                  (b) local_search (relocate / swap / 2-opt* / 2-opt)
                  (c) still overloaded? double the penalty, fix_overload, back to (b)
                Keep the best one.                       (first 15% of the time)
  2. IMPROVE    Ruin and recreate, repeated until the time is up:
                  (a) ruin:          remove a customer and its nearest neighbours
                  (b) recreate:      put each back at its cheapest position
                  (c) search_around: local search on the customers that moved
                                     (same moves as local_search, plus Or-opt)
                  (d) fix_overload:  still overloaded? targeted repair
                  (e) keep the new solution if it is at least as good
  3. FINISH     Check feasibility with utilities.is_feasible and print.

"Good" is compared as (overload, distance): a solution with less overload is
always better, so the same loop first finds a feasible solution (if the
start was not feasible) and then shortens it.
'''
import argparse
import contextlib
import random
import sys
import time

from utilities import read_instance, is_feasible, calculate_cost, size_class, TIME_LIMITS
from optimisations import (new_search, set_routes, snapshot, reduce_routes, local_search,
                           fix_overload, ruin, recreate, search_around,
                           overload, distance, num_routes, out_of_time, log)

SEED = 42
SAFETY = 0.8              # use 80% of the time limit, leaving room for reading input and printing
START_SHARE = 0.15        # share of the time spent building start solutions
MAX_ROUNDS = 30           # max rounds of "local search, then raise the penalty" per start
MAX_REMOVED = 30          # ruin removes between 2 and this many customers
NO_OVERLOAD = 1e9         # penalty so large that overloading a vehicle is never worth it
REPAIR_MAX_N = 400        # fix_overload is slow: in step 2 only use it up to this many nodes


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
def run_yy_cw_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
                   time_limit=None, seed=SEED, verbose=False):
    '''
    Clarke-Wright starts + ruin-and-recreate until the time budget is used.
    Returns (cost, routes) with routes including the depot.
    '''
    start = time.perf_counter()
    num_customers = len(cust_demands) - 1
    if time_limit is None:
        time_limit = time_budget(num_customers)

    search = new_search(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
                        deadline=start + time_limit, seed=seed, verbose=verbose)
    rng = random.Random(seed)

    def state():
        '''One-line summary of the current solution in the search, for the logs.'''
        return (f'{num_routes(search)} routes, distance {distance(search):.1f}, '
                f'overload {overload(search)}')

    def say(msg):
        log(search, f'[{time.perf_counter() - start:6.2f}s] {msg}')

    say(f'{num_customers} customers, {num_vehicles} vehicles of capacity {vehicle_capacity}, '
        f'fleet {sum(cust_demands) / (num_vehicles * vehicle_capacity):.1%} full, '
        f'time budget {time_limit:.1f}s')

    # quality = (overload, distance): smaller is better, overload counts first
    best, best_quality = None, (float('inf'), float('inf'))

    # ---------------------------------------------------------------- (1) START
    say(f'=== (1) START: Clarke-Wright starts for {START_SHARE * time_limit:.1f}s ===')
    starts = 0
    while starts == 0 or time.perf_counter() < start + START_SHARE * time_limit:
        # classic Clarke-Wright first, then randomised ones
        if starts == 0:
            lam = 1.0
            routes = clarke_wright(cust_demands, dist_matrix, vehicle_capacity)
        else:
            lam = rng.uniform(0.5, 1.5)
            routes = clarke_wright(cust_demands, dist_matrix, vehicle_capacity,
                                   rng=rng, noise=0.1, lam=lam)
        search['penalty'] = search['initial_penalty']
        set_routes(search, routes)
        say(f'start {starts} (lam {lam:.2f}) Clarke-Wright:  {state()}')

        # (1a) too many routes: delete the lightest ones, re-insert their customers
        reduce_routes(search)
        say(f'  (1a) after reduce_routes:     {state()}')

        for rnd in range(MAX_ROUNDS):
            # (1b) local search: relocate / swap / 2-opt* / 2-opt until no move helps
            local_search(search)
            say(f'  (1b) after local_search:      {state()}  (penalty {search["penalty"]:.3g})')

            # (1c) feasible: done. overloaded: raise the penalty and repair, then search again
            if overload(search) == 0 or out_of_time(search):
                break
            search['penalty'] *= 2
            fix_overload(search)
            say(f'  (1c) penalty x2, fix_overload: {state()}')

        # keep the best start
        quality = (overload(search), distance(search))
        if num_routes(search) <= num_vehicles and quality < best_quality:
            best, best_quality = snapshot(search), quality
            say(f'  -> new best start')
        starts += 1
        if out_of_time(search):
            break
    say(f'{starts} starts, best: overload {best_quality[0]}, distance {best_quality[1]:.1f}')

    # ---------------------------------------------------------------- (2) IMPROVE
    say(f'=== (2) IMPROVE: ruin and recreate until {time_limit:.1f}s ===')
    k_max = max(2, min(MAX_REMOVED, num_customers // 3))
    steps = better = equal = worse = 0
    while not out_of_time(search):
        steps += 1
        set_routes(search, best)
        search['penalty'] = NO_OVERLOAD

        # (2a) ruin: remove a random customer and its nearest neighbours
        removed = ruin(search, rng.randint(2, k_max))

        # (2b) recreate: put each back at its cheapest position
        recreate(search, removed)

        # (2c) local search, only around the customers that moved
        search_around(search, removed)

        # (2d) still overloaded (very tight fleets): targeted repair
        if overload(search) > 0 and search['n'] <= REPAIR_MAX_N:
            fix_overload(search)

        # (2e) keep it if it is at least as good as the best
        quality = (overload(search), distance(search))
        if quality < best_quality:
            better += 1
            say(f'step {steps}: removed {len(removed)} customers -> new best: {state()}')
        elif quality == best_quality:
            equal += 1
        else:
            worse += 1
        if quality <= best_quality:
            best, best_quality = snapshot(search), quality

        if steps % 500 == 0:
            say(f'  {steps} steps so far: {better} better, {equal} equal (kept), '
                f'{worse} worse (thrown away)')
    say(f'{steps} ruin-and-recreate steps: {better} better, {equal} equal, {worse} worse')
    say(f'final: overload {best_quality[0]}, distance {best_quality[1]:.1f}')

    routes = with_depot(best)
    return calculate_cost(routes, dist_matrix), routes


def main():
    parser = argparse.ArgumentParser(description='Clarke-Wright + ruin-and-recreate CVRP solver')
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

    cost, routes = run_yy_cw_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
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
