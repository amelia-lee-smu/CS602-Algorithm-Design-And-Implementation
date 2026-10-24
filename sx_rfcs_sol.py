import random
import time

INF = float('inf')
ACCEPT = 0.001   # how much worse (as a fraction of the best cost) a solution may be and still be accepted early on


def nn_tour(matrix, start_node):
    # Nearest neighbour giant tour over all customers (depot excluded),
    # starting from start_node.
    tour = [start_node]
    unvisited = set(range(1, len(matrix)))
    unvisited.remove(start_node)
    curr = start_node
    while unvisited:
        nxt_node = min(unvisited, key=lambda c: matrix[curr][c])
        tour.append(nxt_node)
        unvisited.remove(nxt_node)
        curr = nxt_node
    return tour


def two_opt(matrix, init_tour):
    # Improves the order of customers, measuring cost as 0 -> tour -> 0.
    # Returns a new list (customers only); the input list is not modified.
    path = [0] + list(init_tour) + [0]   # anchor the depot at both ends
    n = len(path)
    improved = True
    while improved:
        improved = False
        for i in range(n - 3):
            for j in range(i + 2, n - 1):
                a, b = path[i], path[i + 1]     # edge a-b
                c, d = path[j], path[j + 1]     # edge c-d
                curr_cost = matrix[a][b] + matrix[c][d]
                new_cost = matrix[a][c] + matrix[b][d]
                if new_cost < curr_cost - 1e-10:
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    improved = True
    return path[1:-1]                    # strip the depot again


def split(tour, instance, matrix, capacity, vehicles):
    # instance: demands, depot first (instance[0] == 0)
    # V[k][j]: cheapest cost to serve the first j customers of the tour using exactly k routes
    # P[k][j]: start position i of the last route in that solution (route serves tour[i:j])
    n = len(tour)
    V = [[INF] * (n + 1) for _ in range(vehicles + 1)]
    P = [[-1] * (n + 1) for _ in range(vehicles + 1)]
    V[0][0] = 0

    for k in range(vehicles):
        for i in range(n):
            if V[k][i] == INF:
                continue
            load = 0
            cost = 0
            prev = 0                              # every route starts at the depot
            for idx in range(i, n):               # idx is a list index into tour
                c = tour[idx]
                load += instance[c]
                if load > capacity:
                    break                         # longer chunks only get heavier
                cost += matrix[prev][c]           # leg from previous stop to c
                prev = c
                total = V[k][i] + cost + matrix[c][0]   # plus return to depot
                if total < V[k + 1][idx + 1]:
                    V[k + 1][idx + 1] = total
                    P[k + 1][idx + 1] = i

    # best number of vehicles: cheapest finished cell in the last column
    best_k = min(range(1, vehicles + 1), key=lambda k: V[k][n])
    if V[best_k][n] == INF:
        return None                               # can't split within the vehicle limit

    routes, j, k = [], n, best_k
    while j > 0:
        i = P[k][j]
        routes.append([0] + tour[i:j] + [0])
        j, k = i, k - 1
    return V[best_k][n], routes[::-1]


def route_cost(route, matrix):
    cost = 0
    for loc in range(1, len(route)):
        cost += matrix[route[loc - 1]][route[loc]]
    return cost


# ---------------------------------------------------------------------------
# Helpers for moving customers between routes
# ---------------------------------------------------------------------------

def insertion_cost(route, pos, c, matrix):
    # Extra distance from inserting customer c between route[pos-1] and route[pos]
    a, b = route[pos - 1], route[pos]
    return matrix[a][c] + matrix[c][b] - matrix[a][b]


def build_neighbours(matrix, k=30):
    # For each customer, its k nearest other customers (depot excluded).
    # Moves are only tried between a customer and its neighbours, which keeps
    # the local search fast on large instances.
    n = len(matrix)
    neighbours = [[] for _ in range(n)]
    for c in range(1, n):
        row = matrix[c]
        others = sorted((row[o], o) for o in range(1, n) if o != c)
        neighbours[c] = [o for _, o in others[:k]]
    return neighbours


def local_search(routes, instance, matrix, capacity, neighbours,
                 penalty=None, deadline=None, focus=None):
    # Moves customers between routes to shorten the total distance.
    #   relocate: move customer c next to one of its neighbours in another route
    #   swap:     exchange c with one of its neighbours in another route
    # penalty=None  -> only moves that keep every route within capacity
    # penalty=value -> routes may go over capacity, at `penalty` per unit over
    #                  (used to repair overloaded routes)
    # deadline      -> time.perf_counter() value at which to stop early
    # focus         -> only try moving these customers (default: all)
    # Returns (routes, total_excess_load).
    routes = [list(r) for r in routes]
    loads = [sum(instance[c] for c in r[1:-1]) for r in routes]
    route_of = {}
    for ri, r in enumerate(routes):
        for c in r[1:-1]:
            route_of[c] = ri

    def excess(load):
        return max(0, load - capacity)

    def load_change_cost(old_a, new_a, old_b, new_b):
        if penalty is None:
            return 0 if new_a <= capacity and new_b <= capacity else INF
        return penalty * (excess(new_a) + excess(new_b) - excess(old_a) - excess(old_b))

    customers = list(range(1, len(instance))) if focus is None else list(focus)
    improved = True
    while improved:
        improved = False
        for c in customers:
            if deadline is not None and time.perf_counter() > deadline:
                improved = False
                break
            a = route_of[c]
            ra = routes[a]
            i = ra.index(c)
            prev_c, next_c = ra[i - 1], ra[i + 1]
            removal_gain = matrix[prev_c][c] + matrix[c][next_c] - matrix[prev_c][next_c]

            best = None   # (delta, kind, b, position_or_index)
            for nb in neighbours[c]:
                b = route_of[nb]
                if b == a:
                    continue
                rb = routes[b]
                j = rb.index(nb)

                # relocate c just before or just after nb
                lc = load_change_cost(loads[a], loads[a] - instance[c],
                                      loads[b], loads[b] + instance[c])
                if lc < INF:
                    for pos in (j, j + 1):
                        delta = insertion_cost(rb, pos, c, matrix) - removal_gain + lc
                        if delta < -1e-9 and (best is None or delta < best[0]):
                            best = (delta, 'relocate', b, pos)

                # swap c and nb
                lc = load_change_cost(loads[a], loads[a] - instance[c] + instance[nb],
                                      loads[b], loads[b] - instance[nb] + instance[c])
                if lc < INF:
                    pb, nxb = rb[j - 1], rb[j + 1]
                    old = (matrix[prev_c][c] + matrix[c][next_c] +
                           matrix[pb][nb] + matrix[nb][nxb])
                    new = (matrix[prev_c][nb] + matrix[nb][next_c] +
                           matrix[pb][c] + matrix[c][nxb])
                    delta = new - old + lc
                    if delta < -1e-9 and (best is None or delta < best[0]):
                        best = (delta, 'swap', b, j)

            if best is None:
                continue
            _, kind, b, pos = best
            if kind == 'relocate':
                routes[a].pop(i)
                routes[b].insert(pos, c)
                loads[a] -= instance[c]
                loads[b] += instance[c]
                route_of[c] = b
            else:
                nb = routes[b][pos]
                routes[a][i], routes[b][pos] = nb, c
                loads[a] += instance[nb] - instance[c]
                loads[b] += instance[c] - instance[nb]
                route_of[c], route_of[nb] = b, a
            improved = True

    total_excess = sum(excess(l) for l in loads)
    routes = [[0] + two_opt(matrix, r[1:-1]) + [0] for r in routes if len(r) > 2]
    return routes, total_excess


def reduce_routes(routes, instance, matrix, capacity, vehicles):
    # Removes routes until there are at most `vehicles`, by moving each customer of a
    # route into the cheapest position in another route that still has room.
    # Returns None if it cannot get down to the limit.
    routes = [list(r) for r in routes]
    while len(routes) > vehicles:
        loads = [sum(instance[c] for c in r[1:-1]) for r in routes]
        removed = False
        for idx in sorted(range(len(routes)), key=lambda k: loads[k]):   # lightest first
            others = [list(r) for k, r in enumerate(routes) if k != idx]
            other_loads = [loads[k] for k in range(len(routes)) if k != idx]
            ok = True
            for c in sorted(routes[idx][1:-1], key=lambda c: -instance[c]):  # biggest first
                best = None
                for ri, r in enumerate(others):
                    if other_loads[ri] + instance[c] > capacity:
                        continue
                    for pos in range(1, len(r)):
                        delta = insertion_cost(r, pos, c, matrix)
                        if best is None or delta < best[0]:
                            best = (delta, ri, pos)
                if best is None:
                    ok = False
                    break
                _, ri, pos = best
                others[ri].insert(pos, c)
                other_loads[ri] += instance[c]
            if ok:
                routes = others
                removed = True
                break
        if not removed:
            return None
    return routes


def split_soft(tour, instance, matrix, capacity, vehicles, penalty):
    # Like split, but a route may exceed capacity (up to 2x) at a cost of
    # penalty per unit of excess load. Always uses at most `vehicles` routes.
    n = len(tour)
    V = [[INF] * (n + 1) for _ in range(vehicles + 1)]
    P = [[-1] * (n + 1) for _ in range(vehicles + 1)]
    V[0][0] = 0
    for k in range(vehicles):
        for i in range(n):
            if V[k][i] == INF:
                continue
            load = cost = 0
            prev = 0
            for idx in range(i, n):
                c = tour[idx]
                load += instance[c]
                if load > 2 * capacity:
                    break
                cost += matrix[prev][c]
                prev = c
                total = V[k][i] + cost + matrix[c][0] + penalty * max(0, load - capacity)
                if total < V[k + 1][idx + 1]:
                    V[k + 1][idx + 1] = total
                    P[k + 1][idx + 1] = i
    best_k = min(range(1, vehicles + 1), key=lambda k: V[k][n])
    if V[best_k][n] == INF:
        return None
    routes, j, k = [], n, best_k
    while j > 0:
        i = P[k][j]
        routes.append([0] + tour[i:j] + [0])
        j, k = i, k - 1
    return routes[::-1]


def first_fit_decreasing(instance, matrix, capacity, vehicles):
    # Last-resort feasible solution: pack customers into vehicles by demand,
    # biggest first, ignoring distance. Returns None if they don't fit.
    bins, loads = [[] for _ in range(vehicles)], [0] * vehicles
    for c in sorted(range(1, len(instance)), key=lambda c: -instance[c]):
        for b in range(vehicles):
            if loads[b] + instance[c] <= capacity:
                bins[b].append(c)
                loads[b] += instance[c]
                break
        else:
            return None
    return [[0] + two_opt(matrix, b) + [0] for b in bins if b]


# ---------------------------------------------------------------------------
# Ruin and recreate: uses the time left over after the main search
# ---------------------------------------------------------------------------

def ruin_recreate(routes, instance, matrix, capacity, vehicles, neighbours, rng):
    # Ruin: remove a random customer and a group of customers near it.
    # Recreate: put each removed customer back at its cheapest position that
    # fits, opening a new route if one is still available.
    # Returns (new_routes, removed_customers), or (None, None) if they don't fit.
    n = len(instance) - 1
    seed_c = rng.randint(1, n)
    size = rng.randint(min(3, n), max(3, min(30, n // 4)))
    pool = neighbours[seed_c][:2 * size]
    removed = {seed_c, *rng.sample(pool, min(size - 1, len(pool)))}

    new_routes = []
    for r in routes:
        kept = [c for c in r if c not in removed]
        if len(kept) > 2:
            new_routes.append(kept)
    loads = [sum(instance[c] for c in r[1:-1]) for r in new_routes]

    order = list(removed)
    if rng.random() < 0.5:
        rng.shuffle(order)
    else:
        order.sort(key=lambda c: -instance[c])        # biggest first

    for c in order:
        best = None
        for ri, r in enumerate(new_routes):
            if loads[ri] + instance[c] > capacity:
                continue
            for pos in range(1, len(r)):
                delta = insertion_cost(r, pos, c, matrix)
                if best is None or delta < best[0]:
                    best = (delta, ri, pos)
        if len(new_routes) < vehicles:
            delta = 2 * matrix[0][c]
            if best is None or delta < best[0]:
                best = (delta, None, None)
        if best is None:
            return None, None
        _, ri, pos = best
        if ri is None:
            new_routes.append([0, c, 0])
            loads.append(instance[c])
        else:
            new_routes[ri].insert(pos, c)
            loads[ri] += instance[c]
    return new_routes, removed


def improve_until(best, instance, matrix, capacity, vehicles, neighbours,
                  end, rng, max_iterations=None):
    # Repeats ruin -> recreate -> local search until `end` (or max_iterations).
    # Accepts slightly worse solutions early on so the search can move around,
    # becoming stricter as time runs out. Always returns the best found.
    best_cost, best_routes = best
    cur_cost, cur_routes = best
    start = time.perf_counter()
    iteration = 0
    while True:
        now = time.perf_counter()
        if end is not None and now >= end:
            break
        if max_iterations is not None and iteration >= max_iterations:
            break
        iteration += 1

        new_routes, removed = ruin_recreate(cur_routes, instance, matrix, capacity,
                                            vehicles, neighbours, rng)
        if new_routes is None:
            continue
        focus = set(removed)
        for r in new_routes:
            if any(c in removed for c in r):
                focus.update(r[1:-1])
        new_routes, _ = local_search(new_routes, instance, matrix, capacity,
                                     neighbours, deadline=end, focus=focus)
        new_cost = sum(route_cost(r, matrix) for r in new_routes)

        if end is not None:
            progress = (now - start) / max(1e-9, end - start)
        else:
            progress = iteration / max_iterations
        threshold = ACCEPT * best_cost * (1 - progress)
        if new_cost < cur_cost + threshold:
            cur_cost, cur_routes = new_cost, new_routes
            if new_cost < best_cost - 1e-9:
                best_cost, best_routes = new_cost, new_routes
    return best_cost, best_routes


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def feasible_routes_from_tour(tour, instance, matrix, capacity, vehicles, neighbours,
                              deadline=None):
    # Turns a tour that the normal split could not fit into feasible routes.
    # Cheap repair first, then the penalty-based repair for very tight instances.
    unlimited = split(tour, instance, matrix, capacity, len(instance) - 1)
    if unlimited is not None:
        routes = reduce_routes(unlimited[1], instance, matrix, capacity, vehicles)
        if routes is not None:
            return routes
    for penalty in (10, 100, 1000):
        soft_routes = split_soft(tour, instance, matrix, capacity, vehicles, penalty)
        if soft_routes is None:
            return None
        soft_routes, excess = local_search(soft_routes, instance, matrix, capacity,
                                           neighbours, penalty=penalty, deadline=deadline)
        if excess == 0:
            return soft_routes
    return None


def best_over_starts(instance, matrix, capacity, vehicles, time_limit=None,
                     budget=0.85, cheap_share=0.3, seed=None, rr_iterations=2000):
    # Phase 1 (cheap): for starting customers in random order, build a nearest
    #   neighbour tour, 2-opt it and split it. Uses up to `cheap_share` of the
    #   time budget.
    # Phase 2 (expensive): run the local search between routes on the results,
    #   cheapest first, until the budget runs out. Small instances get through
    #   all of them; large ones only the best few.
    # Phase 3: ruin and recreate on the best solution with whatever time is left.
    # time_limit: seconds allowed for the instance (30 / 60 / 180). The solver
    #   aims to finish within budget * time_limit. None = no limit, in which case
    #   phase 3 runs for rr_iterations iterations.
    t0 = time.perf_counter()
    if time_limit is None:
        end = cheap_end = None
    else:
        end = t0 + budget * time_limit
        cheap_end = t0 + cheap_share * budget * time_limit

    rng = random.Random(seed)
    neighbours = build_neighbours(matrix)
    starts = list(range(1, len(matrix)))
    rng.shuffle(starts)

    candidates = []      # (split cost, routes)
    failed_tours = []    # tours the normal split could not fit
    tried = 0
    for start in starts:
        if cheap_end is not None and tried > 0:
            now = time.perf_counter()
            if now + (now - t0) / tried > cheap_end:
                break
        tour = two_opt(matrix, nn_tour(matrix, start))
        tried += 1
        result = split(tour, instance, matrix, capacity, vehicles)
        if result is not None:
            candidates.append(result)
        else:
            failed_tours.append(tour)

    # time by which phases 1 and 2 must be done, leaving the rest for phase 3
    phase2_end = None if end is None else t0 + 0.6 * budget * time_limit

    # no tour could be split within the vehicle limit: repair a few of them
    if not candidates:
        for tour in failed_tours:
            routes = feasible_routes_from_tour(tour, instance, matrix, capacity,
                                               vehicles, neighbours, deadline=phase2_end)
            if routes is not None:
                candidates.append((sum(route_cost(r, matrix) for r in routes), routes))
            if len(candidates) >= 3:
                break
            if phase2_end is not None and time.perf_counter() > phase2_end:
                break

    # last resort: pack customers by demand
    if not candidates:
        routes = first_fit_decreasing(instance, matrix, capacity, vehicles)
        if routes is None:
            return None
        candidates.append((sum(route_cost(r, matrix) for r in routes), routes))

    # phase 2: improve the best results, cheapest first
    best = None
    for _, routes in sorted(candidates, key=lambda x: x[0]):
        if best is not None and phase2_end is not None and time.perf_counter() > phase2_end:
            break
        improved, _ = local_search(routes, instance, matrix, capacity, neighbours,
                                   deadline=phase2_end)
        cost = sum(route_cost(r, matrix) for r in improved)
        if best is None or cost < best[0]:
            best = (cost, improved)

    # phase 3: ruin and recreate with the remaining time
    return improve_until(best, instance, matrix, capacity, vehicles, neighbours,
                         end, rng, max_iterations=None if end is not None else rr_iterations)