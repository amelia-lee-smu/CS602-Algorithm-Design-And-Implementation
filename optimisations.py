import random
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
# Penalised local search for CVRP (used by clarke_wright_solution.py)
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


class PenalisedLocalSearch:
    '''
    Improves a set of CVRP routes with neighbourhood moves:
        relocate  move one customer to another position or route
        swap      exchange two customers in different routes
        2-opt*    exchange the tails of two routes
        2-opt     reverse a stretch inside one route
    Only each customer's `num_neighbours` nearest customers are tried,
    which keeps it fast on large instances.
    '''

    def __init__(self, cust_demands, dist_matrix, vehicle_capacity, num_vehicles,
                 deadline, num_neighbours=30, seed=42, verbose=False):
        self.demands = cust_demands
        self.dist = dist_matrix
        self.capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.n = len(cust_demands)                  # depot + customers
        self.deadline = deadline                    # time.perf_counter() value to stop at
        self.rng = random.Random(seed)
        self.verbose = verbose

        # each customer's nearest other customers (index 0 = depot, unused)
        k = min(num_neighbours, self.n - 2)
        self.neighbours = [[]] + [
            sorted((j for j in range(1, self.n) if j != c),
                   key=dist_matrix[c].__getitem__)[:k]
            for c in range(1, self.n)
        ]

        # initial penalty per unit of overload (rule of thumb from HGS-CVRP)
        max_dist = max(max(row) for row in dist_matrix)
        self.initial_penalty = max(0.1, min(1000.0, max_dist / max(1, max(cust_demands))))
        self.penalty = self.initial_penalty

    # ------------------------------------------------------------ bookkeeping
    def set_routes(self, routes):
        '''Load a solution and build the lookup tables.'''
        self.routes = [list(r) for r in routes if r]
        self.route_of = [-1] * self.n               # route index of each customer
        self.position = [0] * self.n                # position inside its route
        self.load = [0] * len(self.routes)          # total demand of each route
        self.prefix_load = [[] for _ in self.routes]  # running load along each route
        for i in range(len(self.routes)):
            self.reindex(i)

    def reindex(self, i):
        '''Refresh the lookup tables of route i after it changed.'''
        route = self.routes[i]
        acc, prefix, demands = 0, [0] * len(route), self.demands
        for p, c in enumerate(route):
            self.route_of[c] = i
            self.position[c] = p
            acc += demands[c]
            prefix[p] = acc
        self.prefix_load[i] = prefix
        self.load[i] = acc

    def pred(self, c):
        '''Node visited just before customer c (0 = depot).'''
        p = self.position[c]
        return self.routes[self.route_of[c]][p - 1] if p > 0 else 0

    def succ(self, c):
        '''Node visited just after customer c (0 = depot).'''
        route = self.routes[self.route_of[c]]
        p = self.position[c] + 1
        return route[p] if p < len(route) else 0

    def pen(self, load):
        '''Penalty for a route carrying `load`.'''
        return self.penalty * (load - self.capacity) if load > self.capacity else 0.0

    def num_routes(self):
        return sum(1 for r in self.routes if r)

    def distance(self):
        d, total = self.dist, 0.0
        for r in self.routes:
            if r:
                total += d[0][r[0]] + d[r[-1]][0]
                for a, b in zip(r, r[1:]):
                    total += d[a][b]
        return total

    def overload(self):
        return sum(max(0, l - self.capacity) for l in self.load)

    def out_of_time(self):
        return time.perf_counter() > self.deadline

    def snapshot(self):
        return [list(r) for r in self.routes if r]

    def log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------ fleet repair
    def best_insertion(self, c, skip):
        '''Cheapest (penalised) place to insert customer c, not in route `skip`.
        Returns (cost, route index, position).'''
        d, dc = self.dist, self.demands[c]
        best = (float('inf'), None, None)
        for j, route in enumerate(self.routes):
            if j == skip or not route:
                continue
            extra = self.pen(self.load[j] + dc) - self.pen(self.load[j])
            prev = 0
            for p in range(len(route) + 1):
                nxt = route[p] if p < len(route) else 0
                cost = d[prev][c] + d[c][nxt] - d[prev][nxt] + extra
                if cost < best[0]:
                    best = (cost, j, p)
                prev = nxt
        return best

    def reduce_routes(self):
        '''While there are more routes than vehicles: delete the lightest route
        and re-insert its customers at their cheapest (penalised) positions.
        Some vehicles may end up overloaded; local search fixes that later.'''
        while self.num_routes() > self.num_vehicles:
            i = min((j for j, r in enumerate(self.routes) if r),
                    key=lambda j: self.load[j])
            customers = self.routes[i]
            self.routes[i] = []
            self.reindex(i)
            # biggest demands first: they are the hardest to place
            for c in sorted(customers, key=lambda c: -self.demands[c]):
                _, j, p = self.best_insertion(c, skip=i)
                self.routes[j].insert(p, c)
                self.reindex(j)

    def fix_overload(self):
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
        d, dem = self.dist, self.demands
        progress = False
        while self.overload() > 0 and not self.out_of_time():
            best = (-EPS, None)
            for ru, r1 in enumerate(self.routes):
                if self.load[ru] <= self.capacity:
                    continue
                for u in r1:
                    pu, su, du = self.pred(u), self.succ(u), dem[u]
                    removal = d[pu][u] + d[u][su] - d[pu][su]
                    pen_u = self.pen(self.load[ru] - du) - self.pen(self.load[ru])

                    # (a) give u an unused vehicle of its own
                    if self.num_routes() < self.num_vehicles:
                        delta = d[0][u] + d[u][0] - removal + pen_u
                        if delta < best[0]:
                            best = (delta, ('new_route', u))

                    for rv, r2 in enumerate(self.routes):
                        if rv == ru or not r2:
                            continue
                        Lv = self.load[rv]

                        # (b) relocate u into route rv at its cheapest position
                        extra = pen_u + self.pen(Lv + du) - self.pen(Lv) - removal
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
                            delta = (self.pen(self.load[ru] + diff) - self.pen(self.load[ru])
                                     + self.pen(Lv - diff) - self.pen(Lv))
                            if delta >= 0:
                                continue        # does not help capacity, skip early
                            pv, sv = self.pred(v), self.succ(v)
                            delta += (d[pu][v] + d[v][su] - d[pu][u] - d[u][su]
                                      + d[pv][u] + d[u][sv] - d[pv][v] - d[v][sv])
                            if delta < best[0]:
                                best = (delta, ('swap', u, v))

            move = best[1]
            if move is None:
                break
            self._apply_repair_move(move)
            progress = True
        return progress

    def _apply_repair_move(self, move):
        if move[0] == 'new_route':
            u = move[1]
            ru = self.route_of[u]
            self.routes[ru].pop(self.position[u])
            self.reindex(ru)
            empty = [i for i, r in enumerate(self.routes) if not r]
            if empty:
                rv = empty[0]
            else:
                self.routes.append([])
                self.load.append(0)
                self.prefix_load.append([])
                rv = len(self.routes) - 1
            self.routes[rv] = [u]
            self.reindex(rv)
        elif move[0] == 'relocate':
            _, u, rv, p = move
            ru = self.route_of[u]
            self.routes[ru].pop(self.position[u])
            self.routes[rv].insert(p, u)
            self.reindex(ru)
            self.reindex(rv)
        else:  # swap
            _, u, v = move
            ru, rv = self.route_of[u], self.route_of[v]
            self.routes[ru][self.position[u]] = v
            self.routes[rv][self.position[v]] = u
            self.reindex(ru)
            self.reindex(rv)

    # ------------------------------------------------------------ moves
    # Each try_* computes the change in penalised cost and applies the move
    # only if it is an improvement. Returns True if the move was applied.

    def try_relocate(self, u, v):
        '''Move u next to v (just after v, or just before v).'''
        d, ru, rv = self.dist, self.route_of[u], self.route_of[v]
        pu, su = self.pred(u), self.succ(u)
        removal = d[pu][u] + d[u][su] - d[pu][su]
        cap_delta = 0.0
        if ru != rv:
            du = self.demands[u]
            cap_delta = (self.pen(self.load[ru] - du) - self.pen(self.load[ru])
                         + self.pen(self.load[rv] + du) - self.pen(self.load[rv]))
        for after, (a, b) in ((True, (v, self.succ(v))), (False, (self.pred(v), v))):
            if a == u or b == u:
                continue
            delta = d[a][u] + d[u][b] - d[a][b] - removal + cap_delta
            if delta < -EPS:
                self.routes[ru].pop(self.position[u])
                route = self.routes[rv]
                idx = route.index(v)
                route.insert(idx + 1 if after else idx, u)
                self.reindex(ru)
                if rv != ru:
                    self.reindex(rv)
                return True
        return False

    def try_swap(self, u, v):
        '''Exchange u and v (in different routes).'''
        d, ru, rv, dem = self.dist, self.route_of[u], self.route_of[v], self.demands
        pu, su, pv, sv = self.pred(u), self.succ(u), self.pred(v), self.succ(v)
        delta = (d[pu][v] + d[v][su] - d[pu][u] - d[u][su]
                 + d[pv][u] + d[u][sv] - d[pv][v] - d[v][sv])
        diff = dem[v] - dem[u]
        delta += (self.pen(self.load[ru] + diff) - self.pen(self.load[ru])
                  + self.pen(self.load[rv] - diff) - self.pen(self.load[rv]))
        if delta < -EPS:
            self.routes[ru][self.position[u]] = v
            self.routes[rv][self.position[v]] = u
            self.reindex(ru)
            self.reindex(rv)
            return True
        return False

    def try_2opt_star(self, u, v):
        '''Exchange the tails of the routes of u and v (in different routes).'''
        d, ru, rv = self.dist, self.route_of[u], self.route_of[v]
        su, sv = self.succ(u), self.succ(v)
        r1, r2 = self.routes[ru], self.routes[rv]
        iu, iv = self.position[u], self.position[v]
        L1, L2 = self.load[ru], self.load[rv]
        p1, p2 = self.prefix_load[ru][iu], self.prefix_load[rv][iv]
        base = d[u][su] + d[v][sv]
        old_pen = self.pen(L1) + self.pen(L2)

        # (a)  ...u -> sv...   and   ...v -> su...
        delta = (d[u][sv] + d[v][su] - base
                 + self.pen(p1 + L2 - p2) + self.pen(p2 + L1 - p1) - old_pen)
        if delta < -EPS:
            self.routes[ru] = r1[:iu + 1] + r2[iv + 1:]
            self.routes[rv] = r2[:iv + 1] + r1[iu + 1:]
            self.reindex(ru)
            self.reindex(rv)
            return True

        # (b)  ...u -> v... (r2's head reversed)   and   su... -> sv... (r1's tail reversed)
        delta = (d[u][v] + d[su][sv] - base
                 + self.pen(p1 + p2) + self.pen(L1 - p1 + L2 - p2) - old_pen)
        if delta < -EPS:
            self.routes[ru] = r1[:iu + 1] + r2[:iv + 1][::-1]
            self.routes[rv] = r1[iu + 1:][::-1] + r2[iv + 1:]
            self.reindex(ru)
            self.reindex(rv)
            return True
        return False

    def try_2opt(self, u, v):
        '''Reverse the segment between u and v inside one route.'''
        if self.position[u] > self.position[v]:
            u, v = v, u
        su, sv = self.succ(u), self.succ(v)
        if su == v:
            return False
        d = self.dist
        delta = d[u][v] + d[su][sv] - d[u][su] - d[v][sv]
        if delta < -EPS:
            ri = self.route_of[u]
            route = self.routes[ri]
            i, j = self.position[u] + 1, self.position[v]
            route[i:j + 1] = route[i:j + 1][::-1]
            self.reindex(ri)
            return True
        return False

    def local_search(self):
        '''Apply improving moves until none is left (or time runs out).'''
        customers = list(range(1, self.n))
        while True:
            improved = False
            self.rng.shuffle(customers)
            for u in customers:
                if self.out_of_time():
                    return
                for v in self.neighbours[u]:
                    if self.route_of[u] == self.route_of[v]:
                        moved = self.try_relocate(u, v) or self.try_2opt(u, v)
                    else:
                        moved = (self.try_relocate(u, v) or self.try_swap(u, v)
                                 or self.try_2opt_star(u, v))
                    if moved:
                        improved = True
                        break
            if not improved:
                return

    # ------------------------------------------------------------ driver
    def improve(self, start_routes, max_rounds=30):
        '''
        Full improvement pipeline from a starting solution:
        (1) cut the number of routes down to the number of vehicles
        (2) repeat up to max_rounds:
              local search; if no vehicle is overloaded, stop.
              otherwise double the penalty and run the targeted repair.
        Returns the best feasible routes found, or the last routes if none
        was feasible (the caller should check feasibility).
        '''
        self.penalty = self.initial_penalty
        self.set_routes(start_routes)
        best, best_cost = None, float('inf')
        if self.num_routes() <= self.num_vehicles and self.overload() == 0:
            best, best_cost = self.snapshot(), self.distance()
        self.log(f'  start: {self.num_routes()} routes, cost {self.distance():.1f}')

        self.reduce_routes()
        self.log(f'  after fleet repair: {self.num_routes()} routes, '
                 f'overload {self.overload()}, cost {self.distance():.1f}')

        for rnd in range(max_rounds):
            self.local_search()
            over = self.overload()
            self.log(f'  round {rnd}: penalty={self.penalty:.3g} '
                     f'cost {self.distance():.1f} overload {over}')
            if over == 0:
                cost = self.distance()
                if cost < best_cost:
                    best, best_cost = self.snapshot(), cost
                break
            if self.out_of_time():
                break
            self.penalty *= 2
            if self.fix_overload():
                self.log(f'    targeted repair -> overload {self.overload()}')

        return best if best is not None else self.snapshot()
