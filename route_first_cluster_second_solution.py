import random
import time

from utilities import *
from optimisations import two_opt, three_opt, new_search, improve
from test_cases.sample_from_instructions import *

SEED = 42
INF = float('inf')


def cycle_to_route(cycle):
    '''
    Rotates a tour so that it starts and ends at the depot.
    '''
    start = cycle.index(0)
    return cycle[start:] + cycle[:start] + [0]


def partition_giant_route(tour, cust_demands, dist_matrix, vehicle_capacity):
    '''
    Partition the giant tour by:
    (1) building every possible route between customers, taking into account the vehicle capacity
    (2) find the routes with the lowest cost
    '''
    num_customers = len(tour)

    # adding a new vehicle costs extra distance.
    # we add a large constant when a new vehicle is used, so that the algo uses the least number of vehicles possible.
    # this value has to be more than any possible savings in distance: any solution has at most 2 x num_customers edges, each at most max_dist
    max_dist = max(max(row) for row in dist_matrix)
    new_vehicle_fixed_cost = 2 * num_customers * max_dist + 1           # +1 to make it strictly larger, rather than equal

    # (1) build every possible sub-route that doesnt exceed vehicle capacity
    # possible_routes has the format: (start, length): (cost, route)
    # start = start customers's position in the tour list
    # length = number of consecutive customers the route serves (wraps around end of the tour if needed).
    # cost = distance of the route including depot + new_vehicle_fixed_cost
    # route = the actual route including the depot, e.g. [0, 1, 3, 0]
    possible_routes = {}
    for start in range(num_customers):
        curr_load = 0

        for length in range(1, num_customers + 1):
            customer = tour[(start + length - 1) % num_customers]
            curr_load += cust_demands[customer]

            # dont include routes for consideration if it exceeds the vehicle capacity
            if curr_load > vehicle_capacity:
                break

            customers = [tour[(start + p) % num_customers] for p in range(length)]          # gets the list of customers in the route (wraps ard tour if needed)
            route = cycle_to_route(two_opt([0] + customers, dist_matrix))                   # gets the route
            cost = calculate_cost([route], dist_matrix) + new_vehicle_fixed_cost            # route cost + new_vehicle_fixed_cost
            possible_routes[(start, length)] = (cost, route)

    # (2) find the routes with the lowest cost
    best_cost = INF         # replace when a lower cost is found
    best_routes = None      # replace when a better set of routes are found

    for cut_index in range(num_customers):                          # cut_index = index of the loop where it is cut open to insert the depot
        min_cost_to_serve = [0] + [INF] * (num_customers)           # min_cost_to_serve[i] = min cost to serve the first i customers
        last_route_start = [-1] * (num_customers + 1)               # last_route_start[i] = start of the last route in the min cost path to i

        for route_start in range(num_customers):                                        # route_start: cut point on the straightened line where a new route starts
            start_tour_position = (cut_index + route_start) % num_customers            # start_tour_position : index in tour of the first customer after cut point route_start       
            for length in range(1, num_customers - route_start + 1):                    # length: number of consecutive customers served by the route starting at route_start

                # skip if it wasnt considered a valid sub-route previously
                if (start_tour_position, length) not in possible_routes:
                    break

                # get the new cost
                arc_cost = possible_routes[(start_tour_position, length)][0]            # extract the previously computed cost for this subroute
                new_cost = min_cost_to_serve[route_start] + arc_cost

                # if there is a lower cost way to reach route_start + length (via this route), save it,
                # and record route_start as the start of the last route in last_route_start
                if new_cost < min_cost_to_serve[route_start + length]:
                    min_cost_to_serve[route_start + length] = new_cost
                    last_route_start[route_start + length] = route_start

        # min_cost_to_serve[num_customers] is now the cheapest way to serve every customer for this cut_index.
        # if it beats every cut_index tried so far, save the cost and rebuild its routes
        if min_cost_to_serve[num_customers] < best_cost:
            best_cost = min_cost_to_serve[num_customers]
            best_routes = []

            # rebuild the routes by walking backwards
            route_end = num_customers
            while route_end > 0:
                route_start = last_route_start[route_end]                           # the route ending at route_end started at cut point route_start
                start_tour_position = (cut_index + route_start) % num_customers     # index in tour of this route's first customer
                best_routes.append(possible_routes[(start_tour_position, route_end - route_start)][1])  # look up that route in possible_routes: (tour position, number of customers)
                route_end = route_start                                             # step back to where this route started

    return best_routes


def cheapest_insertion(group, customer, dist_matrix):
    '''
    Finds the cheapest place to insert a customer into a route (group = the route's customers, without the depot).
    Returns (extra distance, index in group to insert at).
    '''
    stops = [0] + group + [0]
    best_extra, best_index = INF, 0
    for p in range(len(stops) - 1):
        a, b = stops[p], stops[p + 1]
        extra = dist_matrix[a][customer] + dist_matrix[customer][b] - dist_matrix[a][b]
        if extra < best_extra:
            best_extra, best_index = extra, p
    return (best_extra, best_index)


def removal_saving(group, index, dist_matrix):
    '''
    Distance saved by removing group[index] from a route (group = the route's customers, without the depot).
    '''
    stops = [0] + group + [0]
    a, c, b = stops[index], stops[index + 1], stops[index + 2]
    return dist_matrix[a][c] + dist_matrix[c][b] - dist_matrix[a][b]


def fit_routes_to_vehicles(routes, cust_demands, dist_matrix, vehicle_capacity, num_vehicles, random_generator, max_moves=2000):
    '''
    Repairs a solution that uses more routes than num_vehicles:
    (1) remove the lightest routes until only num_vehicles routes are left
    (2) insert their customers into the remaining routes at the cheapest position (overloading a route if needed)
    (3) move or swap customers out of overloaded routes until every route is within vehicle_capacity
    Returns the repaired routes, or None if it could not remove all the overload.
    '''
    def overload(load):
        return max(0, load - vehicle_capacity)

    groups = [route[1:-1] for route in routes]                      # customers of each route, without the depot
    loads = [sum(cust_demands[c] for c in group) for group in groups]

    # (1) remove the lightest routes until only num_vehicles are left
    removed = []
    while len(groups) > num_vehicles:
        lightest = loads.index(min(loads))
        removed += groups.pop(lightest)
        loads.pop(lightest)

    # (2) insert each removed customer (biggest demand first) into the route where it causes the least overload,
    # breaking ties by the cheapest extra distance
    for c in sorted(removed, key=lambda c: -cust_demands[c]):
        best = None
        for g in range(len(groups)):
            new_overload = overload(loads[g] + cust_demands[c])
            extra, index = cheapest_insertion(groups[g], c, dist_matrix)
            if best is None or (new_overload, extra) < best[0]:
                best = ((new_overload, extra), g, index)
        _, g, index = best
        groups[g].insert(index, c)
        loads[g] += cust_demands[c]

    # (3) remove the overload, one move at a time.
    # each step takes the move that reduces the total overload the most, even if it reduces it by 0 or makes it
    # slightly worse, so the search can shuffle customers around when no move helps straight away.
    # customers that were just moved are "tabu" (not allowed to move) for a few steps, so the search doesnt undo its own moves.
    tabu_until = {}                                                 # customer -> step until which it cannot be moved
    for step in range(max_moves):
        total_overload = sum(overload(load) for load in loads)
        if total_overload == 0:
            break

        def allowed(customer, overload_change):
            # tabu customers can still move if the move removes all the overload
            return tabu_until.get(customer, -1) < step or total_overload + overload_change == 0

        # find the moves with the best overload change: (kind, g, i, h, j)
        best_change = INF
        best_moves = []
        for g in range(len(groups)):
            if loads[g] <= vehicle_capacity:
                continue
            for i, c in enumerate(groups[g]):
                for h in range(len(groups)):
                    if h == g:
                        continue
                    before = overload(loads[g]) + overload(loads[h])

                    # relocate: move c from route g to route h
                    change = overload(loads[g] - cust_demands[c]) + overload(loads[h] + cust_demands[c]) - before
                    if change <= best_change and allowed(c, change):
                        if change < best_change:
                            best_change, best_moves = change, []
                        best_moves.append(('move', g, i, h, None))

                    # swap: exchange c (in g) with e (in h)
                    for j, e in enumerate(groups[h]):
                        diff = cust_demands[c] - cust_demands[e]
                        if diff == 0:
                            continue
                        change = overload(loads[g] - diff) + overload(loads[h] + diff) - before
                        if change <= best_change and allowed(c, change) and allowed(e, change):
                            if change < best_change:
                                best_change, best_moves = change, []
                            best_moves.append(('swap', g, i, h, j))

        # every customer that could move is tabu, so the repair failed
        if not best_moves:
            return None

        # among the moves with the best overload change, pick the one that adds the least distance
        # (only check a random sample if there are many, to keep this fast)
        if len(best_moves) > 50:
            best_moves = random_generator.sample(best_moves, 50)
        best_distance, best = INF, None
        for kind, g, i, h, j in best_moves:
            c = groups[g][i]
            if kind == 'move':
                distance_change = cheapest_insertion(groups[h], c, dist_matrix)[0] - removal_saving(groups[g], i, dist_matrix)
            else:
                e = groups[h][j]
                extra_c = cheapest_insertion(groups[h][:j] + groups[h][j + 1:], c, dist_matrix)[0]
                extra_e = cheapest_insertion(groups[g][:i] + groups[g][i + 1:], e, dist_matrix)[0]
                distance_change = extra_c + extra_e - removal_saving(groups[g], i, dist_matrix) - removal_saving(groups[h], j, dist_matrix)
            if distance_change < best_distance:
                best_distance, best = distance_change, (kind, g, i, h, j)

        # apply the move, and make the moved customers tabu for a few steps
        kind, g, i, h, j = best
        c = groups[g].pop(i)
        if kind == 'move':
            groups[h].insert(cheapest_insertion(groups[h], c, dist_matrix)[1], c)
            loads[g] -= cust_demands[c]
            loads[h] += cust_demands[c]
        else:
            e = groups[h].pop(j)
            groups[h].insert(cheapest_insertion(groups[h], c, dist_matrix)[1], c)
            groups[g].insert(cheapest_insertion(groups[g], e, dist_matrix)[1], e)
            loads[g] += cust_demands[e] - cust_demands[c]
            loads[h] += cust_demands[c] - cust_demands[e]
            tabu_until[e] = step + random_generator.randint(5, 15)
        tabu_until[c] = step + random_generator.randint(5, 15)

    if any(load > vehicle_capacity for load in loads):
        return None

    # turn the groups back into routes, dropping any route left empty
    return [[0] + group + [0] for group in groups if group]


def is_within_limits(routes, cust_demands, vehicle_capacity, num_vehicles):
    '''
    Quietly checks that there are at most num_vehicles routes and no route is over capacity.
    '''
    if len(routes) > num_vehicles:
        return False
    return all(sum(cust_demands[c] for c in route[1:-1]) <= vehicle_capacity for route in routes)


def improve_between_routes(routes, search, cust_demands, vehicle_capacity, num_vehicles):
    '''
    Improves the routes with moves between routes (relocate, swap, 2-opt*) and inside routes (2-opt),
    using the penalised local search (improve) from optimisations.py.
    Returns the improved routes, or the original routes if the result is not within the vehicle and capacity limits.
    '''
    improved = improve(search, [route[1:-1] for route in routes])          # optimisations.py uses routes without the depot
    improved = [[0] + route + [0] for route in improved if route]

    if is_within_limits(improved, cust_demands, vehicle_capacity, num_vehicles):
        return improved
    return routes


def run_rfcs_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, num_giant_tours=25, seed=SEED, time_limit=None):
    '''
    Implementation of the Route first, cluster second algorithm (Beasley, 1983).
    For each sample of giant tour:
    (1) generate a random giant tour (excluding depot) and improve it with 2-opt
    (2) partition the giant tour into individual routes
    (3) improve the routes with moves between routes (relocate, swap, 2-opt*)
    (4) do 3-opt to optimize each individual route
    '''
    best_cost = INF                                     # replace when a lower cost is found
    best_routes = None                                  # replace when a better set of routes are found
    random_generator = random.Random(seed)                           # random generator with seed
    customers = list(range(1, len(cust_demands)))       # get the customer indexes without the depot

    # local search between routes. it stops early once the deadline is reached (no deadline if time_limit is None)
    deadline = time.perf_counter() + time_limit if time_limit else INF
    search = new_search(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, deadline, seed=seed)

    for _ in range(num_giant_tours):

        # (1) route first: random giant tour, improved with 2-opt
        tour = customers[:]                             # make a copy of the customer list so that it wont be shuffled
        random_generator.shuffle(tour)                               # shuffle the customer order
        tour = two_opt(tour, dist_matrix)               # improve the giant tour using 2-opt

        # (2) cluster second: partition the giant tour
        routes = partition_giant_route(tour, cust_demands, dist_matrix, vehicle_capacity)

        # if the partition uses too many vehicles, repair it. skip this giant tour if the repair fails
        if len(routes) > num_vehicles:
            routes = fit_routes_to_vehicles(routes, cust_demands, dist_matrix, vehicle_capacity, num_vehicles, random_generator)
            if routes is None:
                continue

        # (3) improve the routes with moves between routes
        routes = improve_between_routes(routes, search, cust_demands, vehicle_capacity, num_vehicles)

        # (4) 3-optimise each route
        routes = [cycle_to_route(three_opt(route[:-1], dist_matrix)) for route in routes]

        # store if the cost is lower
        cost = calculate_cost(routes, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_routes = routes

    return (best_cost, best_routes)


# # run the algo
# cost, routes = run_rfcs_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles)
# feasibility_check = is_feasible(routes, cust_demands, vehicle_capacity, num_vehicles)
# print(cost)
# for route in routes:
#     print(" ".join(str(v) for v in route))
# print("feasible:", feasibility_check)
