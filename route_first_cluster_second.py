'''
Route first, cluster second (Beasley, 1983), with modifications.
'''

import heapq
import random

from optimisations import optimised_two_opt, relocate_and_swap, repair_overloads, ruin_and_recreate, two_opt
from utilities import calculate_cost, instance_infeasibility, is_feasible
from utilities import failure_detail, log_checkpoint, open_log  # LOG


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

NUM_NEIGHBOURS = 30          # nearest neighbours used by 2-opt and relocate/swap
NN_CHOICES = 3               # giant tour picks at random among the 3 nearest unvisited customers

SEED = 42                    # random seed, so every run can be reproduced

MAX_OVERLOAD_FACTOR = 1.5    # split_with_vehicle_overload lets a route carry up to 1.5 x vehicle_cap
MAX_PENALTY_DOUBLINGS = 5    # repair doubles the overload penalty at most 5 times before giving up

MIN_REMOVE = 5               # ruin removes a random number of nearby customers,
MAX_REMOVE = 30              # from MIN_REMOVE to min(MAX_REMOVE, num_customers // 4)
NEW_TOUR_PROBABILITY = 0.0   # chance that an iteration builds a new giant tour instead of ruin and recreate

def build_nearest_neighbours(dist_matrix, num_neighbours=NUM_NEIGHBOURS):
    '''
    Finds each customer's num_neighbours closest customers (depot excluded).
    Returns a list where nearest_neighbours[customer] holds that customer's closest customers, nearest first.
    '''
    customers = range(1, len(dist_matrix))
    nearest_neighbours = [[]]  # depot has no neighbours, leave first list empty

    for customer in customers:
        distances_from_customer = dist_matrix[customer]
        # heapq.nsmallest finds the closest few without sorting every customer
        # num_neighbours + 1, as the list includes the customer itself (distance 0)
        closest = heapq.nsmallest(num_neighbours + 1, customers, key=distances_from_customer.__getitem__)

        # remove the customer by value, in case there is another customer at distance 0
        nearest_neighbours.append([other_customer for other_customer in closest if other_customer != customer])

    return nearest_neighbours


# TODO: unchecked
def starting_overload_penalty(cust_demands, dist_matrix):
    '''
    Finds the cost per unit of overload used by split_with_vehicle_overload and repair_overloads:
    average distance between customers / average demand, so one unit of overload costs about one edge.
    Returns the penalty
    '''
    num_customers = len(dist_matrix) - 1
    total_demand = sum(cust_demands)
    if num_customers < 2 or total_demand == 0:
        return 1.0      # no pairs of customers, or no demand to overload with

    # sum each customer's row in C (fast), skipping the depot column
    total_distance = sum(sum(dist_matrix[customer][1:]) for customer in range(1, num_customers + 1))
    average_distance = total_distance / (num_customers * (num_customers - 1))   # diagonal is 0
    average_demand = total_demand / num_customers

    return average_distance / average_demand


# ---------------------------------------------------------------------------
# Giant tour
# ---------------------------------------------------------------------------

def randomised_nn_tour(dist_matrix, nearest_neighbours, rng):
    '''
    Builds a giant tour through all customers (depot excluded):
    - starts from a random customer
    - randomly picks the next customer within the current customer's top NN_CHOICES
    Returns the tour as a list of customers, e.g. [3, 1, 2]
    '''
    num_customers = len(dist_matrix) - 1
    visited = bytearray(num_customers + 1)                 # visited[customer] = 1 once it is in the tour
    unvisited = list(range(1, num_customers + 1))          # only needed when all of a customer's neighbours are visited, remove from list once visited
    index_in_unvisited = list(range(-1, num_customers))    # index_in_unvisited[customer] = its index in unvisited, so visit() can remove it without searching
    tour = []

    def visit(customer):
        # remove customer from unvisited without shifting the list: move the last customer into its place
        index = index_in_unvisited[customer]
        last_customer = unvisited[-1]
        unvisited[index] = last_customer
        index_in_unvisited[last_customer] = index
        unvisited.pop()
        visited[customer] = 1
        tour.append(customer)

    # randomly pick a customer to start the tour
    current_customer = rng.randrange(1, num_customers + 1)
    visit(current_customer)

    while unvisited:            # continues until the unvisited list is empty
        # collect up to NN_CHOICES unvisited customers, taken from the nearest neighbours list
        candidates = []
        for neighbour in nearest_neighbours[current_customer]:
            if not visited[neighbour]:
                candidates.append(neighbour)
                if len(candidates) == NN_CHOICES:
                    break

        # choose the next customer
        if candidates:
            # select from the top nearest unvisited neighbours if any
            current_customer = rng.choice(candidates)
        else:
            # if all the nearest neighbours have been visited,
            # go to the nearest unvisited customer
            distances_from_current = dist_matrix[current_customer]
            current_customer = min(unvisited, key=distances_from_current.__getitem__)

        visit(current_customer)

    return tour


def build_giant_tour(dist_matrix, nearest_neighbours, rng):
    '''
    Builds a randomised nearest neighbour tour, then improves it with 2-opt.
    Returns the tour as a list of customers (depot excluded), e.g. [3, 1, 2]
    '''
    tour = randomised_nn_tour(dist_matrix, nearest_neighbours, rng)

    # 2-opt on the tour
    optimised_tour = optimised_two_opt(tour, dist_matrix, nearest_neighbours)

    return optimised_tour


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
# TODO: unchecked
def split_within_vehicle_capacity(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles):
    '''
    Cuts the giant tour into at most num_vehicles routes at the lowest total cost, keeping the customers in tour order.
    Returns the routes, e.g. [[0, 3, 1, 0], [0, 2, 0]], or None if no split fits within num_vehicles
    '''
    num_customers = len(tour)
    infinity = float("inf")

    # Precompute along the tour, so the cost of any route tour[start..end] is
    # route_start_cost[start] + route_end_cost[end], and extending a route by one customer is one addition
    demands = [cust_demands[customer] for customer in tour]
    distance_along_tour = [0] * num_customers       # distance from tour[0] to tour[position] along the tour
    for position in range(1, num_customers):
        distance_along_tour[position] = (distance_along_tour[position - 1]
                                         + dist_matrix[tour[position - 1]][tour[position]])
    route_start_cost = [dist_matrix[0][customer] - distance_along_tour[position]
                        for position, customer in enumerate(tour)]
    route_end_cost = [distance_along_tour[position] + dist_matrix[customer][0]
                      for position, customer in enumerate(tour)]

    # 1. Split with no vehicle limit (fast): cheapest[end] = cheapest cost to serve tour[0:end]
    cheapest = [0] + [infinity] * num_customers
    route_start = [0] * (num_customers + 1)         # route_start[end] = where the last route ends at `end` starts
    routes_used = [0] * (num_customers + 1)

    for start in range(num_customers):
        if cheapest[start] == infinity:
            continue
        base_cost = cheapest[start] + route_start_cost[start]
        load = 0
        for end in range(start, num_customers):
            load += demands[end]
            if load > vehicle_cap:
                break   # longer routes from this start only get heavier
            total = base_cost + route_end_cost[end]
            if total < cheapest[end + 1]:
                cheapest[end + 1] = total
                route_start[end + 1] = start
                routes_used[end + 1] = routes_used[start] + 1

    if cheapest[num_customers] == infinity:
        return None     # a customer's demand alone is over vehicle_cap

    # if it already fits within num_vehicles, it is also the best split with the limit
    if routes_used[num_customers] <= num_vehicles:
        # rebuild the routes by following route_start back from the end of the tour
        routes = []
        end = num_customers
        while end > 0:
            start = route_start[end]
            routes.append([0, *tour[start:end], 0])
            end = start
        return routes[::-1]

    # 2. Too many routes: split with the vehicle limit
    # cheapest_with_k[end] = cheapest cost to serve tour[0:end] with exactly k routes
    demand_left = [0] * (num_customers + 1)         # demand_left[start] = total demand of tour[start:]
    for position in range(num_customers - 1, -1, -1):
        demand_left[position] = demand_left[position + 1] + demands[position]

    cheapest_with_k = [0] + [infinity] * num_customers      # k = 0 routes
    route_starts = [None]                                   # route_starts[k][end], for k = 1, 2, ...
    best_total, best_num_routes = infinity, 0

    for k in range(1, num_vehicles + 1):
        cheapest_next = [infinity] * (num_customers + 1)
        starts_next = [0] * (num_customers + 1)
        max_load_left = (num_vehicles - k + 1) * vehicle_cap  # most demand this route and the remaining ones can carry

        for start in range(num_customers):
            if cheapest_with_k[start] == infinity or demand_left[start] > max_load_left:
                continue    # not reachable with k - 1 routes, or the rest can never fit in the vehicles left
            base_cost = cheapest_with_k[start] + route_start_cost[start]
            load = 0
            for end in range(start, num_customers):
                load += demands[end]
                if load > vehicle_cap:
                    break
                total = base_cost + route_end_cost[end]
                if total < cheapest_next[end + 1]:
                    cheapest_next[end + 1] = total
                    starts_next[end + 1] = start

        route_starts.append(starts_next)
        if cheapest_next[num_customers] < best_total:
            best_total, best_num_routes = cheapest_next[num_customers], k
        cheapest_with_k = cheapest_next

    if best_total == infinity:
        return None     # this tour order cannot be split into num_vehicles routes

    # rebuild the routes by following route_starts back from the end of the tour, one route per k
    routes = []
    end = num_customers
    for k in range(best_num_routes, 0, -1):
        start = route_starts[k][end]
        routes.append([0, *tour[start:end], 0])
        end = start
    return routes[::-1]

# TODO: unchecked
def split_with_vehicle_overload(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles, penalty):
    '''
    Cuts the giant tour into at most num_vehicles routes, keeping the customers in tour order,
    but lets a route go over vehicle_cap (up to MAX_OVERLOAD_FACTOR x vehicle_cap) at a cost of
    penalty per unit of overload. Used when split_within_vehicle_capacity finds no cut that fits.
    Same table as pass 2 of split_within_vehicle_capacity, with route cost = distance + penalty * max(0, load - vehicle_cap).
    Returns the routes (some may be overloaded), or None if even this finds no cut
    '''
    num_customers = len(tour)
    infinity = float("inf")
    max_load = MAX_OVERLOAD_FACTOR * vehicle_cap

    # Precompute along the tour (same as split_within_vehicle_capacity), so the distance of route tour[start..end] is
    # route_start_cost[start] + route_end_cost[end]
    demands = [cust_demands[customer] for customer in tour]
    distance_along_tour = [0] * num_customers
    for position in range(1, num_customers):
        distance_along_tour[position] = (distance_along_tour[position - 1]
                                         + dist_matrix[tour[position - 1]][tour[position]])
    route_start_cost = [dist_matrix[0][customer] - distance_along_tour[position]
                        for position, customer in enumerate(tour)]
    route_end_cost = [distance_along_tour[position] + dist_matrix[customer][0]
                      for position, customer in enumerate(tour)]

    demand_left = [0] * (num_customers + 1)         # demand_left[start] = total demand of tour[start:]
    for position in range(num_customers - 1, -1, -1):
        demand_left[position] = demand_left[position + 1] + demands[position]

    # cheapest_with_k[end] = cheapest distance + penalty to serve tour[0:end] with exactly k routes
    cheapest_with_k = [0] + [infinity] * num_customers      # k = 0 routes
    route_starts = [None]                                   # route_starts[k][end], for k = 1, 2, ...
    best_total, best_num_routes = infinity, 0

    for k in range(1, num_vehicles + 1):
        cheapest_next = [infinity] * (num_customers + 1)
        starts_next = [0] * (num_customers + 1)
        max_load_left = (num_vehicles - k + 1) * max_load  # most demand this route and the remaining ones can carry

        for start in range(num_customers):
            cost_before = cheapest_with_k[start]
            if cost_before == infinity or demand_left[start] > max_load_left:
                continue    # not reachable with k - 1 routes, or the rest can never fit in the vehicles left
            base_cost = cost_before + route_start_cost[start]
            load = 0
            for end in range(start, num_customers):
                load += demands[end]
                if load > max_load:
                    break   # longer routes from this start only get heavier
                total = base_cost + route_end_cost[end]
                if load > vehicle_cap:
                    total += penalty * (load - vehicle_cap)
                if total < cheapest_next[end + 1]:
                    cheapest_next[end + 1] = total
                    starts_next[end + 1] = start

        route_starts.append(starts_next)
        if cheapest_next[num_customers] < best_total:
            best_total, best_num_routes = cheapest_next[num_customers], k
        cheapest_with_k = cheapest_next

    if best_total == infinity:
        return None     # even with overload allowed, this tour cannot be split into num_vehicles routes

    # rebuild the routes by following route_starts back from the end of the tour, one route per k
    routes = []
    end = num_customers
    for k in range(best_num_routes, 0, -1):
        start = route_starts[k][end]
        routes.append([0, *tour[start:end], 0])
        end = start
    return routes[::-1]

# TODO: unchecked
def split_with_overload_and_repair(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                                   nearest_neighbours, rng, penalty):
    '''
    Splits the tour allowing overloaded routes, then moves customers until nothing is overloaded.
    Returns (routes, method, failed_routes):
        (routes, "split_soft", None)                      if it worked
        (None, "split_soft_no_cut", None)                 if there is no cut, even with overload allowed
        (None, "repair_failed", still overloaded routes)  if the overload cannot be removed
    '''
    routes = split_with_vehicle_overload(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles, penalty)
    if routes is None:
        return None, "split_soft_no_cut", None  # no cut, even allowing routes up to MAX_OVERLOAD_FACTOR x vehicle_cap

    # move customers until nothing is overloaded
    routes, repaired = repair_overloads(routes, cust_demands, dist_matrix, vehicle_cap,
                                        nearest_neighbours, rng, penalty, MAX_PENALTY_DOUBLINGS)
    if not repaired:
        return None, "repair_failed", routes
    return routes, "split_soft", None

# TODO: unchecked
def partition_giant_tour(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                         nearest_neighbours, rng, penalty):
    '''
    Cuts the giant tour into feasible routes:
    - tries split_within_vehicle_capacity first
    - if it finds no cut that fits within num_vehicles, uses split_with_overload_and_repair
    The routes are checked with is_feasible before they are returned, so returned routes are always feasible.

    Returns (routes, method, failed_routes). method says which approach worked, or why it failed:
        (routes, "split", None) or (routes, "split_soft", None)   feasible routes
        (None, "split_soft_no_cut", None)                         no cut, even with overload allowed
        (None, "repair_failed", still overloaded routes)          the overload could not be removed
        (None, "partition_infeasible", routes that broke a rule)  failed is_feasible (a bug: both paths
                                                                  should give feasible routes)
    failed_routes is only there so the log can say what went wrong; the solver does not use it.
    '''
    # 1. exact split: fast, no repair needed
    routes = split_within_vehicle_capacity(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles)
    method = "split"

    # 2. no cut fits: split with overload allowed, then repair
    if routes is None:
        routes, method, failed_routes = split_with_overload_and_repair(
            tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, rng, penalty)
        if routes is None:
            return None, method, failed_routes

    # 3. only feasible routes leave this function
    if not is_feasible(routes, cust_demands, vehicle_cap, num_vehicles):
        return None, "partition_infeasible", routes
    return routes, method, None


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
# TODO: unchecked
def checkpoint(routes, state, step):
    '''
    Called after each step: the partition method, relocate_swap and two_opt.
    1. Recompute the cost from the routes (calculate_cost in utilities)
    2. If it is cheaper than the best so far, check it with is_feasible,
       then save it to best["solution"] as one (cost, routes) write
    3. If logging is on (log is not None), write a row to the CSV log (log_checkpoint in utilities)
    state holds: cust_demands, dist_matrix, vehicle_cap, num_vehicles,
    best, best_cost, best_routes, iteration, log
    Returns the cost of routes
    '''
    # 1. recompute the cost from the routes, so it is exactly what the grader will check
    cost = calculate_cost(state["dist_matrix"], routes)

    # 2. save a new best: the cheap cost comparison first, so is_feasible only runs on improvements
    new_best = False  # LOG
    if cost < state["best_cost"] and is_feasible(routes, state["cust_demands"], state["vehicle_cap"], state["num_vehicles"]):
        best_routes = [route[:] for route in routes]   # copy, so later steps cannot change the saved routes
        state["best_cost"] = cost                       # local copy: fast to compare against
        state["best_routes"] = best_routes              # local copy: what ruin and recreate starts from
        state["best"]["solution"] = (cost, best_routes) # shared copy: what run_algo returns, in one write
        new_best = True  # LOG

    # 3. log this step (skipped entirely when logging is off)
    if state["log"] is not None:  # LOG
        log_checkpoint(state["log"], state["iteration"], step, cost, state["best_cost"], new_best,  # LOG
                       len(routes), state["num_vehicles"])  # LOG

    return cost


# ---------------------------------------------------------------------------
# Improvement steps
# ---------------------------------------------------------------------------
# TODO: unchecked
def improve_between_routes(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                           customers_to_check=None):
    '''
    Moves and swaps customers between nearby routes, keeping every route within vehicle_cap.
    customers_to_check: only try moves for these customers (None = every customer).
    Returns the improved routes (a new list; the input routes are not changed)
    '''
    return relocate_and_swap(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                             customers_to_check=customers_to_check)

# TODO: unchecked
def improve_within_routes(routes, dist_matrix, customers_to_check=None):
    '''
    Reorders the customers inside each route with plain 2-opt (loads do not change, so routes stay feasible).
    Plain 2-opt is used, not optimised_two_opt: routes are short, so trying every pair is faster
    and finds more improvements. The routes are changed in place.
    customers_to_check: only 2-opt the routes that contain one of these customers (None = every route).
    Returns the improved routes
    '''
    if customers_to_check is None:
        for route in routes:
            two_opt(route, dist_matrix)
        return routes

    changed_customers = set(customers_to_check)
    for route in routes:
        if not changed_customers.isdisjoint(route):     # this route holds a changed customer
            two_opt(route, dist_matrix)
    return routes

# TODO: unchecked
def improve_feasible_routes(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                            save_checkpoint, customers_to_check=None):
    '''
    Improves feasible routes (from partition_giant_tour, or from ruin and recreate):
    1. between routes: relocate and swap customers between nearby routes
    2. within routes: 2-opt on each route
    customers_to_check: only work around these customers, e.g. the ones ruin and recreate moved
    (None = the whole solution, used after partitioning a new giant tour).
    save_checkpoint(routes, step) is called after each step, so a new best is saved and logged.
    Returns the improved routes
    '''
    routes = improve_between_routes(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                                    customers_to_check)
    save_checkpoint(routes, "relocate_swap")

    routes = improve_within_routes(routes, dist_matrix, customers_to_check)
    save_checkpoint(routes, "two_opt")

    return routes


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_rfcs(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info=None):
    '''
    best:        shared dict from run_algo; new best solutions are saved to best["solution"]
    log_info:    None, or the details for the CSV log (see open_log in utilities)
    '''

    # Setup
    # 1. CSV log (None if logging is off)
    log = open_log(log_info)  # LOG

    # 2. random number generator, so every run can be reproduced
    rng = random.Random(SEED)

    # 3. get the current best cost
    best_cost = best["solution"][0]

    # 4. get each customer's nearest neighbours, so 2-opt and relocate/swap only try nearby customers
    nearest_neighbours = build_nearest_neighbours(dist_matrix, NUM_NEIGHBOURS)

    # 5. cost per unit of overload, for split_with_vehicle_overload and repair_overloads
    overload_penalty = starting_overload_penalty(cust_demands, dist_matrix)

    # 6. how many customers ruin removes: a random number from min_remove to max_remove,
    #    at most a quarter of the customers so small instances are not rebuilt from scratch
    num_customers = len(cust_demands) - 1
    max_remove = max(1, min(MAX_REMOVE, num_customers // 4))
    min_remove = min(MIN_REMOVE, max_remove)

    # 7. everything checkpoint needs, in one dict that is updated in place
    #    (best_cost, best_routes and iteration change during the run)
    state = {
        "cust_demands": cust_demands,
        "dist_matrix": dist_matrix,
        "vehicle_cap": vehicle_cap,
        "num_vehicles": num_vehicles,
        "best": best,
        "best_cost": best_cost,
        "best_routes": best["solution"][1],     # Clarke-Wright's routes if it saved any, else None
        "iteration": 0,
        "log": log,  # LOG
    }

    # checkpoint with state already filled in, so the improvement steps only pass (routes, step)
    def save_checkpoint(routes, step):
        return checkpoint(routes, state, step)

    # one log row for a step that checkpoint does not see (setup, giant tour, failures)
    def log_row(step, cost=None, num_routes=None, detail=""):  # LOG
        log_checkpoint(log, state["iteration"], step, cost, state["best_cost"], False,  # LOG
                       num_routes, num_vehicles, detail)  # LOG

    total_demand = sum(cust_demands)  # LOG
    if log is not None: log_row("start", detail=(  # LOG
        f"seed {SEED}, {num_customers} customers, {num_vehicles} vehicles, capacity {vehicle_cap}, "  # LOG
        f"total demand {total_demand}, tightness {total_demand / (num_vehicles * vehicle_cap):.3f}"))  # LOG

    # 8. stop straight away if no routes could ever be feasible, instead of looping until the time limit
    impossible = instance_infeasibility(cust_demands, vehicle_cap, num_vehicles)
    if impossible:
        if log is not None: log_row("instance_infeasible", detail=impossible)  # LOG
        return

    # Repeat until run_algo terminates the process
    while True:
        state["iteration"] += 1

        # A. no feasible solution yet (or, rarely, for variety): build one from a new giant tour
        if state["best_routes"] is None or rng.random() < NEW_TOUR_PROBABILITY:
            # 1. new randomised giant tour, improved with 2-opt
            tour = build_giant_tour(dist_matrix, nearest_neighbours, rng)
            if log is not None: log_row("giant_tour", cost=calculate_cost(dist_matrix, [[0, *tour, 0]]))  # LOG

            # 2. cut it into feasible routes
            routes, method, failed_routes = partition_giant_tour(tour, cust_demands, dist_matrix, vehicle_cap,
                                                                 num_vehicles, nearest_neighbours, rng,
                                                                 overload_penalty)
            if routes is None:
                if log is not None: log_row(method, num_routes=len(failed_routes) if failed_routes else None,  # LOG
                                            detail=failure_detail(method, failed_routes, cust_demands,  # LOG
                                                                  vehicle_cap, num_vehicles))  # LOG
                continue    # this tour could not be made feasible, try a new one

            # 3. save the partition result if it is a new best (step = "split" or "split_soft")
            save_checkpoint(routes, method)

            # 4. relocate/swap, then 2-opt on each route, saving any new best after each step
            improve_feasible_routes(routes, cust_demands, dist_matrix, vehicle_cap,
                                    nearest_neighbours, rng, save_checkpoint)

        # B. otherwise: ruin and recreate the best solution so far, then improve around the moved customers
        else:
            num_remove = rng.randint(min_remove, max_remove)
            routes, removed = ruin_and_recreate(state["best_routes"], cust_demands, dist_matrix, vehicle_cap,
                                                num_vehicles, nearest_neighbours, num_remove, rng)
            if routes is None:
                if log is not None: log_row("recreate_failed", detail=f"{num_remove} customers removed")  # LOG
                continue    # a removed customer could not be put back, try again

            save_checkpoint(routes, "ruin_recreate")
            improve_feasible_routes(routes, cust_demands, dist_matrix, vehicle_cap,
                                    nearest_neighbours, rng, save_checkpoint, customers_to_check=removed)
