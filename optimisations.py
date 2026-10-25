import random
import sys
import time

EPS = 1e-9

def two_opt(cycle, dist_matrix):
    '''
    Repeatedly replaces edges (a,b) and (c,d) with (a,c) and (b,d) whenever it shortens the tour
    (e.g. uncrossing two crossed edges), until no further improvement can be made.
    '''
    num_customers = len(cycle)
    improved = True

    while improved:
        improved = False
        for i in range(num_customers - 1):
            for j in range(i + 2, num_customers):

                # do nothing if edges are adjacent
                if i == 0 and j == num_customers - 1:
                    continue

                a, b = cycle[i], cycle[i + 1]
                c, d = cycle[j], cycle[(j + 1) % num_customers]         # j might go past the end of the list, so a wrap aroung might be needed
                delta = dist_matrix[a][c] + dist_matrix[b][d] - dist_matrix[a][b] - dist_matrix[c][d]

                # if it reduces the distance, swap the nodes
                if delta < -EPS:
                    cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                    improved = True

    return cycle

def three_opt(cycle, dist_matrix):
    '''
    Repeatedly removes three edges (a,b), (c,d) and (e,f) and tries reconnecting the segments
    (reversing a segment, or swapping two segments), keeping any change that shortens the tour,
    until no further improvement can be made.
    '''
    num_customers = len(cycle)
    dist = dist_matrix
    improved = True

    while improved:
        improved = False
        for i in range(num_customers):
            for j in range(i + 2, num_customers):
                for k in range(j + 2, num_customers + (i > 0)):
                    a, b = cycle[i - 1], cycle[i]
                    c, d = cycle[j - 1], cycle[j]
                    e, f = cycle[k - 1], cycle[k % num_customers]
                    d0 = dist[a][b] + dist[c][d] + dist[e][f]
                    d1 = dist[a][c] + dist[b][d] + dist[e][f]
                    d2 = dist[a][b] + dist[c][e] + dist[d][f]
                    d3 = dist[a][d] + dist[e][b] + dist[c][f]
                    d4 = dist[f][b] + dist[c][d] + dist[e][a]
                    if d1 < d0 - EPS:
                        cycle[i:j] = reversed(cycle[i:j])
                    elif d2 < d0 - EPS:
                        cycle[j:k] = reversed(cycle[j:k])
                    elif d4 < d0 - EPS:
                        cycle[i:k] = reversed(cycle[i:k])
                    elif d3 < d0 - EPS:
                        cycle[i:k] = cycle[j:k] + cycle[i:j]
                    else:
                        continue
                    improved = True

    return cycle


# =============================================================================
# Penalised local search + ruin and recreate for CVRP
# =============================================================================
#
# Routes here are lists of customers WITHOUT the depot, e.g. [3, 1, 2]
# means 0 -> 3 -> 1 -> 2 -> 0.
#
# While searching, a vehicle MAY be overloaded, but every unit of overload
# costs a penalty P. Every decision compares this penalised cost:
#
#     penalised cost = total distance + P * (total overload)
#
# Raising P over time pushes the search back towards feasible solutions.
#
# All functions share one "search" dict, made by new_search(). It holds
#   the problem:   demands, dist, capacity, num_vehicles, n, neighbours
#   the settings:  deadline, rng, verbose, penalty, initial_penalty
#   the solution:  routes, plus lookup tables kept up to date by reindex():
#                  route_of[c], position[c], load[r], prefix_load[r]
#
# Typical use:
#     search = new_search(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, deadline)
#     routes = improve(search, start_routes)                  # penalised local search
#     routes = ruin_and_recreate_until_deadline(search, routes)  # keep improving
#
# Map of what calls what. improve() and ruin_and_recreate_until_deadline() are
# shortcuts; clarke_wright_solution.py and yy_clarke_wright_solution.py write
# the same steps out themselves so the solver shows what is going on, and
# route_first_cluster_second_solution.py calls improve():
#
#   improve(search, routes)                      penalised local search, stops at a local optimum
#   ├── reduce_routes        too many routes? delete the lightest, re-insert its customers
#   │   └── best_insertion       cheapest place for one customer
#   ├── local_search         every customer, until no move helps:
#   │   ├── try_relocate         move 1 customer
#   │   ├── try_swap             swap 2 customers (different routes)
#   │   ├── try_2opt_star        swap the tails of 2 routes
#   │   └── try_2opt             reverse a stretch inside 1 route
#   └── fix_overload         still overloaded? move/swap customers anywhere
#
#   ruin_and_recreate_until_deadline(search, routes)   repeats until the deadline:
#   └── ruin_and_recreate    one step, kept if (overload, distance) is no worse
#       ├── ruin                 remove a customer and its nearest neighbours
#       ├── recreate             put them back
#       │   └── best_insertion
#       ├── search_around        local search on the moved customers only:
#       │   ├── try_relocate, try_swap, try_2opt_star, try_2opt   (as above)
#       │   └── try_or_opt           move a run of 2-3 customers
#       └── fix_overload         (only on instances up to REPAIR_MAX_N nodes)
#
# NOT called by either: two_opt() and three_opt() above. They improve one
# route on its own (route_first_cluster_second_solution.py uses them).
# try_2opt does the same kind of reversal as two_opt, but only between a
# customer and its nearest neighbours, as one move inside the search.


def new_search(cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
               deadline, num_neighbours=30, seed=42, verbose=False):
    '''
    Makes the search dict used by every function below.
    deadline: time.perf_counter() value to stop at.
    Only each customer's `num_neighbours` nearest customers are tried in the
    local search, which keeps it fast on large instances.
    '''
    n = len(cust_demands)                           # depot + customers

    # each customer's nearest other customers (index 0 = depot, unused)
    k = min(num_neighbours, n - 2)
    neighbours = [[]] + [
        sorted((j for j in range(1, n) if j != c), key=dist_matrix[c].__getitem__)[:k]
        for c in range(1, n)
    ]

    # initial penalty per unit of overload (rule of thumb from HGS-CVRP)
    max_dist = max(max(row) for row in dist_matrix)
    initial_penalty = max(0.1, min(1000.0, max_dist / max(1, max(cust_demands))))

    search = {
        'demands': cust_demands, 'dist': dist_matrix, 'capacity': vehicle_capacity,
        'num_vehicles': num_vehicles, 'n': n, 'neighbours': neighbours,
        'deadline': deadline, 'rng': random.Random(seed), 'verbose': verbose,
        'initial_penalty': initial_penalty, 'penalty': initial_penalty,
    }
    set_routes(search, [])
    return search


# ------------------------------------------------------------ bookkeeping
def set_routes(search, routes):
    '''Load a solution into the search and build the lookup tables.'''
    search['routes'] = [list(r) for r in routes if r]
    search['route_of'] = [-1] * search['n']        # route index of each customer
    search['position'] = [0] * search['n']         # position inside its route
    search['load'] = [0] * len(search['routes'])   # total demand of each route
    search['prefix_load'] = [[] for _ in search['routes']]  # running load along each route
    for i in range(len(search['routes'])):
        reindex(search, i)


def reindex(search, i):
    '''Refresh the lookup tables of route i after it changed.'''
    route, demands = search['routes'][i], search['demands']
    route_of, position = search['route_of'], search['position']
    acc, prefix = 0, [0] * len(route)
    for p, c in enumerate(route):
        route_of[c] = i
        position[c] = p
        acc += demands[c]
        prefix[p] = acc
    search['prefix_load'][i] = prefix
    search['load'][i] = acc


def add_empty_route(search):
    '''Index of an empty route, adding one if there is none.'''
    for i, r in enumerate(search['routes']):
        if not r:
            return i
    search['routes'].append([])
    search['load'].append(0)
    search['prefix_load'].append([])
    return len(search['routes']) - 1


def pred(search, c):
    '''Node visited just before customer c (0 = depot).'''
    p = search['position'][c]
    return search['routes'][search['route_of'][c]][p - 1] if p > 0 else 0


def succ(search, c):
    '''Node visited just after customer c (0 = depot).'''
    route = search['routes'][search['route_of'][c]]
    p = search['position'][c] + 1
    return route[p] if p < len(route) else 0


def pen(search, load):
    '''Penalty for a route carrying `load`.'''
    capacity = search['capacity']
    return search['penalty'] * (load - capacity) if load > capacity else 0.0


def num_routes(search):
    return sum(1 for r in search['routes'] if r)


def distance(search):
    d, total = search['dist'], 0.0
    for r in search['routes']:
        if r:
            total += d[0][r[0]] + d[r[-1]][0]
            for a, b in zip(r, r[1:]):
                total += d[a][b]
    return total


def overload(search):
    capacity = search['capacity']
    return sum(max(0, l - capacity) for l in search['load'])


def out_of_time(search):
    return time.perf_counter() > search['deadline']


def snapshot(search):
    '''Copy of the current routes (empty routes dropped).'''
    return [list(r) for r in search['routes'] if r]


def log(search, msg):
    '''Print msg if verbose. Goes to stderr, so stdout keeps only the solution.'''
    if search['verbose']:
        print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------ fleet repair
def best_insertion(search, c, skip):
    '''Cheapest (penalised) place to insert customer c, not in route `skip`.
    Returns (cost, route index, position).'''
    d, dc, load = search['dist'], search['demands'][c], search['load']
    best = (float('inf'), None, None)
    for j, route in enumerate(search['routes']):
        if j == skip or not route:
            continue
        extra = pen(search, load[j] + dc) - pen(search, load[j])
        prev = 0
        for p in range(len(route) + 1):
            nxt = route[p] if p < len(route) else 0
            cost = d[prev][c] + d[c][nxt] - d[prev][nxt] + extra
            if cost < best[0]:
                best = (cost, j, p)
            prev = nxt
    return best


def reduce_routes(search):
    '''While there are more routes than vehicles: delete the lightest route
    and re-insert its customers at their cheapest (penalised) positions.
    Some vehicles may end up overloaded; local search fixes that later.'''
    routes, load, demands = search['routes'], search['load'], search['demands']
    while num_routes(search) > search['num_vehicles']:
        i = min((j for j, r in enumerate(routes) if r), key=lambda j: load[j])
        customers = routes[i]
        routes[i] = []
        reindex(search, i)
        # biggest demands first: they are the hardest to place
        for c in sorted(customers, key=lambda c: -demands[c]):
            _, j, p = best_insertion(search, c, skip=i)
            routes[j].insert(p, c)
            reindex(search, j)


def fix_overload(search):
    '''Targeted repair for when local search is stuck with an overloaded vehicle.

    Local search only tries nearby customers, but fixing a tight fleet can
    need a move between customers that are far apart. For every customer u
    in an overloaded route, try:
      - moving u into an unused vehicle (0 -> u -> 0),
      - relocating u to ANY other route,
      - swapping u with ANY customer in another route,
    and apply the best move that lowers the penalised cost.
    Repeat until there is no overload or no improving move.
    '''
    d, dem, load, capacity = search['dist'], search['demands'], search['load'], search['capacity']
    progress = False
    while overload(search) > 0 and not out_of_time(search):
        best = (-EPS, None)
        for ru, r1 in enumerate(search['routes']):
            if load[ru] <= capacity:
                continue
            for u in r1:
                pu, su, du = pred(search, u), succ(search, u), dem[u]
                removal = d[pu][u] + d[u][su] - d[pu][su]
                pen_u = pen(search, load[ru] - du) - pen(search, load[ru])

                # (a) give u an unused vehicle of its own
                if num_routes(search) < search['num_vehicles']:
                    delta = d[0][u] + d[u][0] - removal + pen_u
                    if delta < best[0]:
                        best = (delta, ('new_route', u))

                for rv, r2 in enumerate(search['routes']):
                    if rv == ru or not r2:
                        continue
                    Lv = load[rv]

                    # (b) relocate u into route rv at its cheapest position
                    extra = pen_u + pen(search, Lv + du) - pen(search, Lv) - removal
                    prev = 0
                    for p in range(len(r2) + 1):
                        nxt = r2[p] if p < len(r2) else 0
                        delta = d[prev][u] + d[u][nxt] - d[prev][nxt] + extra
                        if delta < best[0]:
                            best = (delta, ('relocate', u, rv, p))
                        prev = nxt

                    # (c) swap u with each v in route rv
                    for v in r2:
                        diff = dem[v] - du
                        delta = (pen(search, load[ru] + diff) - pen(search, load[ru])
                                 + pen(search, Lv - diff) - pen(search, Lv))
                        if delta >= 0:
                            continue        # does not help capacity, skip early
                        pv, sv = pred(search, v), succ(search, v)
                        delta += (d[pu][v] + d[v][su] - d[pu][u] - d[u][su]
                                  + d[pv][u] + d[u][sv] - d[pv][v] - d[v][sv])
                        if delta < best[0]:
                            best = (delta, ('swap', u, v))

        move = best[1]
        if move is None:
            break
        apply_repair_move(search, move)
        progress = True
    return progress


def apply_repair_move(search, move):
    routes, route_of, position = search['routes'], search['route_of'], search['position']
    if move[0] == 'new_route':
        u = move[1]
        ru = route_of[u]
        routes[ru].pop(position[u])
        reindex(search, ru)
        rv = add_empty_route(search)
        routes[rv] = [u]
        reindex(search, rv)
    elif move[0] == 'relocate':
        _, u, rv, p = move
        ru = route_of[u]
        routes[ru].pop(position[u])
        routes[rv].insert(p, u)
        reindex(search, ru)
        reindex(search, rv)
    else:  # swap
        _, u, v = move
        ru, rv = route_of[u], route_of[v]
        routes[ru][position[u]] = v
        routes[rv][position[v]] = u
        reindex(search, ru)
        reindex(search, rv)


# ------------------------------------------------------------ moves
# Each try_* computes the change in penalised cost and applies the move
# only if it is an improvement. Returns True if the move was applied.

def try_relocate(search, u, v):
    '''Move u next to v (just after v, or just before v).'''
    d, route_of, load = search['dist'], search['route_of'], search['load']
    ru, rv = route_of[u], route_of[v]
    pu, su = pred(search, u), succ(search, u)
    removal = d[pu][u] + d[u][su] - d[pu][su]
    cap_delta = 0.0
    if ru != rv:
        du = search['demands'][u]
        cap_delta = (pen(search, load[ru] - du) - pen(search, load[ru])
                     + pen(search, load[rv] + du) - pen(search, load[rv]))
    for after, (a, b) in ((True, (v, succ(search, v))), (False, (pred(search, v), v))):
        if a == u or b == u:
            continue
        delta = d[a][u] + d[u][b] - d[a][b] - removal + cap_delta
        if delta < -EPS:
            search['routes'][ru].pop(search['position'][u])
            route = search['routes'][rv]
            idx = route.index(v)
            route.insert(idx + 1 if after else idx, u)
            reindex(search, ru)
            if rv != ru:
                reindex(search, rv)
            return True
    return False


def try_swap(search, u, v):
    '''Exchange u and v (in different routes).'''
    d, dem, load = search['dist'], search['demands'], search['load']
    ru, rv = search['route_of'][u], search['route_of'][v]
    pu, su, pv, sv = pred(search, u), succ(search, u), pred(search, v), succ(search, v)
    delta = (d[pu][v] + d[v][su] - d[pu][u] - d[u][su]
             + d[pv][u] + d[u][sv] - d[pv][v] - d[v][sv])
    diff = dem[v] - dem[u]
    delta += (pen(search, load[ru] + diff) - pen(search, load[ru])
              + pen(search, load[rv] - diff) - pen(search, load[rv]))
    if delta < -EPS:
        search['routes'][ru][search['position'][u]] = v
        search['routes'][rv][search['position'][v]] = u
        reindex(search, ru)
        reindex(search, rv)
        return True
    return False


def try_2opt_star(search, u, v):
    '''Exchange the tails of the routes of u and v (in different routes).'''
    d, routes = search['dist'], search['routes']
    ru, rv = search['route_of'][u], search['route_of'][v]
    su, sv = succ(search, u), succ(search, v)
    r1, r2 = routes[ru], routes[rv]
    iu, iv = search['position'][u], search['position'][v]
    L1, L2 = search['load'][ru], search['load'][rv]
    p1, p2 = search['prefix_load'][ru][iu], search['prefix_load'][rv][iv]
    base = d[u][su] + d[v][sv]
    old_pen = pen(search, L1) + pen(search, L2)

    # (a)  ...u -> sv...   and   ...v -> su...
    delta = (d[u][sv] + d[v][su] - base
             + pen(search, p1 + L2 - p2) + pen(search, p2 + L1 - p1) - old_pen)
    if delta < -EPS:
        routes[ru] = r1[:iu + 1] + r2[iv + 1:]
        routes[rv] = r2[:iv + 1] + r1[iu + 1:]
        reindex(search, ru)
        reindex(search, rv)
        return True

    # (b)  ...u -> v... (r2's head reversed)   and   su... -> sv... (r1's tail reversed)
    delta = (d[u][v] + d[su][sv] - base
             + pen(search, p1 + p2) + pen(search, L1 - p1 + L2 - p2) - old_pen)
    if delta < -EPS:
        routes[ru] = r1[:iu + 1] + r2[:iv + 1][::-1]
        routes[rv] = r1[iu + 1:][::-1] + r2[iv + 1:]
        reindex(search, ru)
        reindex(search, rv)
        return True
    return False


def try_2opt(search, u, v):
    '''Reverse the segment between u and v inside one route.'''
    position = search['position']
    if position[u] > position[v]:
        u, v = v, u
    su, sv = succ(search, u), succ(search, v)
    if su == v:
        return False
    d = search['dist']
    delta = d[u][v] + d[su][sv] - d[u][su] - d[v][sv]
    if delta < -EPS:
        ri = search['route_of'][u]
        route = search['routes'][ri]
        i, j = position[u] + 1, position[v]
        route[i:j + 1] = route[i:j + 1][::-1]
        reindex(search, ri)
        return True
    return False


def try_or_opt(search, u, v):
    '''Move the run of 2 or 3 customers starting at u next to v
    (just after or just before v), keeping or reversing its order.'''
    d, load = search['dist'], search['load']
    ru, rv = search['route_of'][u], search['route_of'][v]
    r1, i = search['routes'][ru], search['position'][u]
    for length in (2, 3):
        if i + length > len(r1):
            return False
        run = r1[i:i + length]
        if v in run:
            continue
        first, last = run[0], run[-1]
        before = r1[i - 1] if i > 0 else 0
        after = r1[i + length] if i + length < len(r1) else 0
        removal = d[before][first] + d[last][after] - d[before][after]
        cap_delta = 0.0
        if ru != rv:
            run_load = sum(search['demands'][c] for c in run)
            cap_delta = (pen(search, load[ru] - run_load) - pen(search, load[ru])
                         + pen(search, load[rv] + run_load) - pen(search, load[rv]))
        for a, b in ((v, succ(search, v)), (pred(search, v), v)):     # insert between a and b
            if a in run or b in run:
                continue
            forward = d[a][first] + d[last][b] - d[a][b] - removal + cap_delta
            backward = d[a][last] + d[first][b] - d[a][b] - removal + cap_delta
            if min(forward, backward) < -EPS:
                del r1[i:i + length]
                route = search['routes'][rv]
                idx = route.index(a) + 1 if a != 0 else 0
                route[idx:idx] = run if forward <= backward else run[::-1]
                reindex(search, ru)
                if rv != ru:
                    reindex(search, rv)
                return True
    return False


# ------------------------------------------------------------ local search
def local_search(search):
    '''Apply improving moves (relocate / swap / 2-opt* / 2-opt) until none
    is left (or time runs out), checking every customer each pass.'''
    customers = list(range(1, search['n']))
    route_of = search['route_of']
    while True:
        improved = False
        search['rng'].shuffle(customers)
        for u in customers:
            if out_of_time(search):
                return
            for v in search['neighbours'][u]:
                if route_of[u] == route_of[v]:
                    moved = try_relocate(search, u, v) or try_2opt(search, u, v)
                else:
                    moved = (try_relocate(search, u, v) or try_swap(search, u, v)
                             or try_2opt_star(search, u, v))
                if moved:
                    improved = True
                    break
        if not improved:
            return


def search_around(search, customers):
    '''Local search that only tries moves for customers in a queue
    (relocate / swap / 2-opt* / 2-opt / Or-opt). After a move, the customers
    next to it go (back) into the queue. Much faster than local_search when
    only a few customers changed.'''
    queue, queued = [], set()
    route_of = search['route_of']

    def add(c):
        if c != 0 and c not in queued:
            queued.add(c)
            queue.append(c)

    for c in customers:
        add(c)
    head = 0
    while head < len(queue) and not out_of_time(search):
        u = queue[head]
        head += 1
        queued.discard(u)
        old_neighbours = (pred(search, u), succ(search, u))
        for v in search['neighbours'][u]:
            if route_of[u] == route_of[v]:
                moved = (try_relocate(search, u, v) or try_2opt(search, u, v)
                         or try_or_opt(search, u, v))
            else:
                moved = (try_relocate(search, u, v) or try_swap(search, u, v)
                         or try_2opt_star(search, u, v) or try_or_opt(search, u, v))
            if moved:
                for c in (u, v, *old_neighbours, pred(search, u), succ(search, u),
                          pred(search, v), succ(search, v)):
                    add(c)
                break


def improve(search, start_routes, max_rounds=30):
    '''
    Penalised local search from a starting solution:
    (1) reduce_routes: cut the number of routes down to the number of vehicles
    (2) repeat up to max_rounds:
          local_search (relocate / swap / 2-opt* / 2-opt);
          if no vehicle is overloaded, stop.
          otherwise double the penalty and run fix_overload.
    Does NOT use two_opt(), three_opt() or try_or_opt.
    Returns the best feasible routes found, or the last routes if none
    was feasible (the caller should check feasibility).
    '''
    search['penalty'] = search['initial_penalty']
    set_routes(search, start_routes)
    best, best_cost = None, float('inf')
    if num_routes(search) <= search['num_vehicles'] and overload(search) == 0:
        best, best_cost = snapshot(search), distance(search)
    log(search, f'  start: {num_routes(search)} routes, cost {distance(search):.1f}')

    reduce_routes(search)
    log(search, f'  after fleet repair: {num_routes(search)} routes, '
                f'overload {overload(search)}, cost {distance(search):.1f}')

    for rnd in range(max_rounds):
        local_search(search)
        over = overload(search)
        log(search, f'  round {rnd}: penalty={search["penalty"]:.3g} '
                    f'cost {distance(search):.1f} overload {over}')
        if over == 0:
            cost = distance(search)
            if cost < best_cost:
                best, best_cost = snapshot(search), cost
            break
        if out_of_time(search):
            break
        search['penalty'] *= 2
        if fix_overload(search):
            log(search, f'    targeted repair -> overload {overload(search)}')

    return best if best is not None else snapshot(search)


# ------------------------------------------------------------ ruin and recreate
# Local search only makes small moves that improve straight away, so it gets
# stuck in a local optimum. Ruin and recreate makes a big change instead
# (remove a group of customers, put them back) and only judges the result.

NO_OVERLOAD = 1e9         # penalty so large that overloading a vehicle is never worth it
REPAIR_MAX_N = 400        # fix_overload is slow, only use it up to this many nodes


def ruin(search, k):
    '''Remove a random customer and its k-1 nearest customers. Returns them.'''
    seed = search['rng'].randrange(1, search['n'])
    removed = [seed] + search['neighbours'][seed][:k - 1]
    gone = set(removed)
    routes = search['routes']
    for r in {search['route_of'][c] for c in removed}:
        routes[r] = [c for c in routes[r] if c not in gone]
        reindex(search, r)
    return removed


def recreate(search, removed):
    '''Put each removed customer (random order) at its cheapest position,
    or in an unused vehicle if that is cheaper.'''
    search['rng'].shuffle(removed)
    for c in removed:
        cost, r, p = best_insertion(search, c, skip=-1)
        if num_routes(search) < search['num_vehicles'] and 2 * search['dist'][0][c] < cost:
            r, p = add_empty_route(search), 0
        search['routes'][r].insert(p, c)
        reindex(search, r)


def ruin_and_recreate(search, routes, k):
    '''One step: ruin k customers, recreate, search around them.
    Returns (overload, distance, routes) of the new solution.'''
    set_routes(search, routes)
    search['penalty'] = NO_OVERLOAD
    removed = ruin(search, k)
    recreate(search, removed)
    search_around(search, removed)
    # still overloaded (very tight fleets): targeted repair, which moves
    # customers anywhere. It is slow, so only on small/medium instances
    if overload(search) > 0 and search['n'] <= REPAIR_MAX_N:
        fix_overload(search)
    return overload(search), distance(search), snapshot(search)


def ruin_and_recreate_until_deadline(search, routes, max_removed=30):
    '''
    Repeat ruin_and_recreate until the deadline, keeping a new solution if
    it is at least as good. "Good" is (overload, distance), overload first,
    so an overloaded start is first made feasible, then shortened.
    routes must use at most num_vehicles routes. Returns the best routes.
    '''
    rng = search['rng']
    set_routes(search, routes)
    best, best_quality = snapshot(search), (overload(search), distance(search))
    k_max = max(2, min(max_removed, (search['n'] - 1) // 3))     # n - 1 customers
    steps = 0
    while not out_of_time(search):
        steps += 1
        over, dist, new = ruin_and_recreate(search, best, rng.randint(2, k_max))
        if (over, dist) <= best_quality:            # at least as good: keep it
            if (over, dist) < best_quality:
                log(search, f'step {steps}: overload {over}, distance {dist:.1f}')
            best, best_quality = new, (over, dist)
    log(search, f'{steps} ruin-and-recreate steps')
    return best
