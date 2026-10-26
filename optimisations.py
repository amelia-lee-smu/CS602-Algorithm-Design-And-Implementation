'''
Improvement steps shared by Clarke-Wright and RFCS.
'''
# TODO: unchecked
def two_opt(path, dist_matrix):
    '''
    Improves the order of a path by reversing segments, trying every pair of edges,
    until no reversal makes it shorter. 
    The first and last items stay fixed. The path is changed in place.
    Returns the improved path.
    '''
    last_index = len(path) - 1

    # fewer than 2 items between the fixed ends: nothing to reverse
    if last_index < 3:
        return path

    improved = True
    while improved:  # stops when there is no more improvement
        improved = False

        # edge (current_customer, next_customer) = (path[current_index], path[current_index + 1])
        for current_index in range(last_index - 1):
            current_customer = path[current_index]
            distances_from_current = dist_matrix[current_customer]
            next_customer = path[current_index + 1]
            distances_from_next = dist_matrix[next_customer]    # read once, not on every check below
            current_edge_length = distances_from_current[next_customer]

            # edge (other_customer, after_other) = (path[other_index], path[other_index + 1]),
            # starting 2 positions on so the two edges never touch
            for other_index in range(current_index + 2, last_index):
                other_customer = path[other_index]
                after_other = path[other_index + 1]

                # replacing (current, next) and (other, after_other)
                # with (current, other) and (next, after_other)
                saving = (current_edge_length + dist_matrix[other_customer][after_other]
                          - distances_from_current[other_customer] - distances_from_next[after_other])
                if saving > 1e-9:   # stops tiny rounding errors in decimal distances
                    # reverse the customers between the two edges
                    path[current_index + 1:other_index + 1] = path[current_index + 1:other_index + 1][::-1]
                    improved = True

                    # the edge from current_customer has changed, so update it and keep going
                    next_customer = path[current_index + 1]
                    distances_from_next = dist_matrix[next_customer]
                    current_edge_length = distances_from_current[next_customer]

    return path


def optimised_two_opt(path, dist_matrix, nearest_neighbours):
    '''
    Similar to the original two_opt algorithm, except each customer is only paired with its nearest neighbours.
    First and last items stay fixed. The path is changed in place.
    Returns the improved path
    '''
    last_index = len(path) - 1

    # fewer than 2 items between the fixed ends: nothing to reverse
    if last_index < 3:
        return path

    # position[customer] = its index in path, so a neighbour can be found without searching
    position = {customer: index for index, customer in enumerate(path)}

    improved = True
    while improved:  # stops when there is no more improvement
        improved = False

        # edge (current_customer, next_customer) = (path[current_index], path[current_index + 1])
        for current_index in range(last_index):
            current_customer = path[current_index]
            next_customer = path[current_index + 1]
            distances_from_current = dist_matrix[current_customer]
            current_edge_length = distances_from_current[next_customer]

            # pair current_customer with each of its nearest neighbours that is in this path
            for neighbour in nearest_neighbours[current_customer]:
                neighbour_index = position.get(neighbour)

                # if neighbour is not in this path, has no next item, or the two edges touch, do nothing
                if (neighbour_index is None or neighbour_index >= last_index
                        or current_index - 1 <= neighbour_index <= current_index + 1):
                    continue

                # edge (neighbour, after_neighbour) = (path[neighbour_index], path[neighbour_index + 1])
                after_neighbour = path[neighbour_index + 1]

                # replacing (current, next) and (neighbour, after_neighbour)
                # with (current, neighbour) and (next, after_neighbour)
                saving = (current_edge_length + dist_matrix[neighbour][after_neighbour]
                          - distances_from_current[neighbour] - dist_matrix[next_customer][after_neighbour])
                if saving > 1e-9: # stops tiny rounding errors in decimal distances
                    # reverse the customers between the two edges
                    if neighbour_index > current_index:
                        start, end = current_index + 1, neighbour_index
                    else:
                        start, end = neighbour_index + 1, current_index
                    path[start:end + 1] = path[start:end + 1][::-1]
                    for index in range(start, end + 1):
                        position[path[index]] = index

                    improved = True
                    break       # next_customer has changed, so move on to the next edge

    return path


# ---------------------------------------------------------------------------
# Swapping customers between routes
# ---------------------------------------------------------------------------
# TODO: unchecked
def relocate_and_swap(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng, penalty=None,
                      customers_to_check=None):
    '''
    Moves customers between routes until no move lowers the cost:
    - relocate: move a customer next to one of its nearest neighbours on another route
    - swap: exchange a customer with one of its nearest neighbours on another route
    penalty=None: every route must stay within vehicle_cap (normal improvement).
    penalty given: routes may be overloaded, and cost = distance + penalty * total overload (repair).
    customers_to_check: only try moves for these customers (e.g. the ones ruin and recreate moved),
    instead of every customer; much faster when only a small part of the solution changed.
    Returns the improved routes (a new list; empty routes are removed)
    '''
    routes = [list(route) for route in routes]      # copy, so the caller's routes are not changed
    loads = [sum(cust_demands[customer] for customer in route) for route in routes]

    # route_of[customer] = index of its route, position_in_route[customer] = its index in that route
    route_of = [0] * len(dist_matrix)
    position_in_route = [0] * len(dist_matrix)

    def index_route(route_index):
        route = routes[route_index]
        for position in range(1, len(route) - 1):   # skip the depot at both ends
            route_of[route[position]] = route_index
            position_in_route[route[position]] = position

    for route_index in range(len(routes)):
        index_route(route_index)

    def overload_cost(load):
        return penalty * (load - vehicle_cap) if load > vehicle_cap else 0

    def try_relocate(customer):
        '''Moves customer to the best spot next to a nearest neighbour on another route. Returns True if moved.'''
        from_route_index = route_of[customer]
        from_route = routes[from_route_index]
        from_position = position_in_route[customer]
        before, after = from_route[from_position - 1], from_route[from_position + 1]
        distances_from_customer = dist_matrix[customer]
        demand = cust_demands[customer]

        # distance saved by taking customer out of its route
        removal_saving = (dist_matrix[before][customer] + distances_from_customer[after]
                          - dist_matrix[before][after])

        best_change, best_move = -1e-9, None        # only accept moves that lower the cost
        for neighbour in nearest_neighbours[customer]:
            to_route_index = route_of[neighbour]
            if to_route_index == from_route_index:
                continue

            # capacity: forbidden without a penalty, extra cost with one
            if penalty is None:
                if loads[to_route_index] + demand > vehicle_cap:
                    continue
                capacity_change = 0
            else:
                capacity_change = (overload_cost(loads[from_route_index] - demand)
                                   - overload_cost(loads[from_route_index])
                                   + overload_cost(loads[to_route_index] + demand)
                                   - overload_cost(loads[to_route_index]))

            # insert just before or just after the neighbour
            to_route = routes[to_route_index]
            neighbour_position = position_in_route[neighbour]
            for insert_position in (neighbour_position, neighbour_position + 1):
                previous_stop, next_stop = to_route[insert_position - 1], to_route[insert_position]
                insertion_cost = (dist_matrix[previous_stop][customer] + distances_from_customer[next_stop]
                                  - dist_matrix[previous_stop][next_stop])
                change = insertion_cost - removal_saving + capacity_change
                if change < best_change:
                    best_change, best_move = change, (to_route_index, insert_position)

        if best_move is None:
            return False

        to_route_index, insert_position = best_move
        from_route.pop(from_position)
        routes[to_route_index].insert(insert_position, customer)
        loads[from_route_index] -= demand
        loads[to_route_index] += demand
        index_route(from_route_index)
        index_route(to_route_index)
        return True

    def try_swap(customer):
        '''Swaps customer with the best nearest neighbour on another route. Returns True if swapped.'''
        route_a_index = route_of[customer]
        route_a = routes[route_a_index]
        position_a = position_in_route[customer]
        before_a, after_a = route_a[position_a - 1], route_a[position_a + 1]
        demand_a = cust_demands[customer]
        distances_from_customer = dist_matrix[customer]
        customer_edges = distances_from_customer[before_a] + distances_from_customer[after_a]

        best_change, best_neighbour = -1e-9, None
        for neighbour in nearest_neighbours[customer]:
            route_b_index = route_of[neighbour]
            if route_b_index == route_a_index:
                continue
            demand_b = cust_demands[neighbour]
            new_load_a = loads[route_a_index] - demand_a + demand_b
            new_load_b = loads[route_b_index] - demand_b + demand_a

            if penalty is None:
                if new_load_a > vehicle_cap or new_load_b > vehicle_cap:
                    continue
                capacity_change = 0
            else:
                capacity_change = (overload_cost(new_load_a) - overload_cost(loads[route_a_index])
                                   + overload_cost(new_load_b) - overload_cost(loads[route_b_index]))

            route_b = routes[route_b_index]
            position_b = position_in_route[neighbour]
            before_b, after_b = route_b[position_b - 1], route_b[position_b + 1]
            distances_from_neighbour = dist_matrix[neighbour]

            # neighbour takes customer's place, and customer takes neighbour's place
            change = (distances_from_neighbour[before_a] + distances_from_neighbour[after_a] - customer_edges
                      + distances_from_customer[before_b] + distances_from_customer[after_b]
                      - distances_from_neighbour[before_b] - distances_from_neighbour[after_b]
                      + capacity_change)
            if change < best_change:
                best_change, best_neighbour = change, neighbour

        if best_neighbour is None:
            return False

        route_b_index = route_of[best_neighbour]
        position_b = position_in_route[best_neighbour]
        route_a[position_a] = best_neighbour
        routes[route_b_index][position_b] = customer
        demand_b = cust_demands[best_neighbour]
        loads[route_a_index] += demand_b - demand_a
        loads[route_b_index] += demand_a - demand_b
        route_of[customer], route_of[best_neighbour] = route_b_index, route_a_index
        position_in_route[customer], position_in_route[best_neighbour] = position_b, position_a
        return True

    # repeat passes over the customers, in a new random order each time, until a pass makes no move
    if customers_to_check is None:
        customers = [customer for route in routes for customer in route[1:-1]]
    else:
        customers = list(customers_to_check)
    improved = True
    while improved:
        improved = False
        rng.shuffle(customers)
        for customer in customers:
            if try_relocate(customer) or try_swap(customer):
                improved = True

    return [route for route in routes if len(route) > 2]    # drop routes left empty



# ---------------------------------------------------------------------------
# Partitioning of routes
# ---------------------------------------------------------------------------
MAX_OVERLOAD_FACTOR = 1.5    # split_with_vehicle_overload lets a route carry up to 1.5 x vehicle_cap
MAX_PENALTY_DOUBLINGS = 5    # repair doubles the overload penalty at most 5 times before giving up

# TODO: unchecked
def split_within_vehicle_capacity(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles):
    '''
    Cuts the giant tour into at most num_vehicles routes at the lowest total cost, keeping the customers in tour order.
    Returns the routes, or None if no split fits within num_vehicles for this tour order.
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
    Cuts the giant tour into at most num_vehicles routes, keeping the customers in tour order.
    Each route may go over vehicle_cap (up to MAX_OVERLOAD_FACTOR x vehicle_cap) at a cost of penalty per unit of overload. 
    Used when split_within_vehicle_capacity finds no cut that fits.
    Returns the (overloaded) routes , or None if even this finds no cuts.
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
def repair_overloads(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                     penalty, max_doublings):
    '''
    Fixes the overloads left by split_with_vehicle_overload.
    Repeat up to max_doublings + 1 times:
    1. relocate_and_swap with the penalty, until no move lowers distance + penalty * overload
    2. If no route is over vehicle_cap, stop: the routes are feasible
    3. Otherwise double the penalty, so overload counts for more in the next round
    Returns the feasible routes, or None if a route is still overloaded at the end
    '''
    def has_overload(routes):
        for route in routes:
            load = 0
            for customer in route:
                load += cust_demands[customer]
            if load > vehicle_cap:
                return True
        return False

    for _ in range(max_doublings + 1):
        routes = relocate_and_swap(routes, cust_demands, dist_matrix, vehicle_cap,
                                   nearest_neighbours, rng, penalty)
        if not has_overload(routes):
            return routes
        penalty *= 2    # local to this call: the caller's starting penalty is unchanged

    return None     # still overloaded after every doubling


# TODO: unchecked
def split_with_overload_and_repair(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                                   nearest_neighbours, rng, penalty):
    '''
    Splits the tour by allowing overloaded routes first, then moves customers until there is no overload.
    Returns (routes, method):
        (routes, "split_soft")             if it worked
        (None, "split_soft_no_cut")        if there is no cut, even with overload allowed
        (None, "repair_failed")            if the overload cannot be removed
    '''
    routes = split_with_vehicle_overload(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles, penalty)
    if routes is None:
        return None, "split_soft_no_cut"    # no cut, even allowing routes up to MAX_OVERLOAD_FACTOR x vehicle_cap

    # move customers until nothing is overloaded
    routes = repair_overloads(routes, cust_demands, dist_matrix, vehicle_cap,
                              nearest_neighbours, rng, penalty, MAX_PENALTY_DOUBLINGS)
    if routes is None:
        return None, "repair_failed"
    
    return routes, "split_soft"


# ---------------------------------------------------------------------------
# Reducing the number of vehicles used
# ---------------------------------------------------------------------------
# TODO: unchecked
def vehicle_reduction(routes, cust_demands, num_vehicles):
    '''
    Removes the lightest routes until only num_vehicles are left.
    The input routes are not changed.
    Returns (kept routes, customers of the removed routes)
    '''
    num_to_remove = len(routes) - num_vehicles
    if num_to_remove <= 0:
        return [list(route) for route in routes], []

    # route indexes from lightest to heaviest; the lightest have the fewest demand units to place elsewhere
    loads = [sum(cust_demands[customer] for customer in route) for route in routes]
    lightest_first = sorted(range(len(routes)), key=loads.__getitem__)
    removed_route_indexes = set(lightest_first[:num_to_remove])

    kept_routes = [list(route) for index, route in enumerate(routes) if index not in removed_route_indexes]
    removed_customers = [customer for index in removed_route_indexes for customer in routes[index][1:-1]]
    return kept_routes, removed_customers


# TODO: unchecked
def cheapest_insertion(routes, removed_customers, cust_demands, dist_matrix, vehicle_cap):
    '''
    Inserts each removed customer (biggest demand first) into the route where it causes the least overload,
    breaking ties by the cheapest extra distance. Routes may end up overloaded.
    Returns the updated routes.
    '''
    loads = [sum(cust_demands[customer] for customer in route) for route in routes]

    # biggest demand first: the hardest customers to fit get placed while there is the most room left
    for customer in sorted(removed_customers, key=cust_demands.__getitem__, reverse=True):
        demand = cust_demands[customer]
        distances_from_customer = dist_matrix[customer]
        best_overload, best_extra_distance = float("inf"), float("inf")
        best_route_index, best_position = None, None

        for route_index, route in enumerate(routes):
            # extra overload this insertion would cause on this route (0 if it still fits)
            load = loads[route_index]
            extra_overload = max(0, load + demand - vehicle_cap) - max(0, load - vehicle_cap)
            if extra_overload > best_overload:
                continue    # worse than the best route so far, whatever the distance

            for position in range(1, len(route)):
                previous_stop, next_stop = route[position - 1], route[position]
                extra_distance = (dist_matrix[previous_stop][customer] + distances_from_customer[next_stop]
                                  - dist_matrix[previous_stop][next_stop])
                if (extra_overload, extra_distance) < (best_overload, best_extra_distance):
                    best_overload, best_extra_distance = extra_overload, extra_distance
                    best_route_index, best_position = route_index, position

        routes[best_route_index].insert(best_position, customer)
        loads[best_route_index] += demand

    return routes



# ---------------------------------------------------------------------------
# Ruin and recreate
# ---------------------------------------------------------------------------
MIN_REMOVE = 5                      # ruin removes a random number of nearby customers,
MAX_REMOVE = 30                     # from MIN_REMOVE to min(MAX_REMOVE, num_customers // 4)


# TODO: unchecked
def ruin(routes, nearest_neighbours, num_remove, rng):
    '''
    Removes a group of customers that are close together:
    a random customer, plus its nearest neighbours until num_remove customers are removed.
    Routes left with no customers are dropped. The input routes are not changed.
    Returns (ruined_routes, removed_customers)
    '''
    customers = [customer for route in routes for customer in route[1:-1]]
    first_removed = rng.choice(customers)
    removed_customers = [first_removed, *nearest_neighbours[first_removed][:num_remove - 1]]

    removed_set = set(removed_customers)
    ruined_routes = []
    for route in routes:
        kept_route = [stop for stop in route if stop not in removed_set]     # the depot (0) is never removed
        if len(kept_route) > 2:
            ruined_routes.append(kept_route)
    return ruined_routes, removed_customers


# TODO: unchecked
def recreate(routes, removed_customers, cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, rng):
    '''
    Puts the removed customers back one at a time (in random order), each at its cheapest spot, in a route that still has room:
    1. first try the spots just before or after its nearest neighbours (fast, and usually the cheapest)
    2. if none of those routes have room, try every spot in every route with room
    3. if it still fits nowhere, open a new route [0, customer, 0] while vehicles are left
    The routes are changed in place.
    Returns the routes, or None if a customer could not be placed
    '''
    loads = [sum(cust_demands[customer] for customer in route) for route in routes]

    # route_of[customer] = index of its route (-1 while it is not in a route), position_in_route = its index
    route_of = [-1] * len(dist_matrix)
    position_in_route = [0] * len(dist_matrix)

    def index_route(route_index):
        route = routes[route_index]
        for position in range(1, len(route) - 1):
            route_of[route[position]] = route_index
            position_in_route[route[position]] = position

    for route_index in range(len(routes)):
        index_route(route_index)

    def insertion_cost(route, position, distances_from_customer):
        # extra distance from putting the customer between route[position - 1] and route[position]
        previous_stop, next_stop = route[position - 1], route[position]
        return (dist_matrix[previous_stop][customer] + distances_from_customer[next_stop]
                - dist_matrix[previous_stop][next_stop])

    insert_order = list(removed_customers)
    rng.shuffle(insert_order)       # a different order each time gives different rebuilds

    for customer in insert_order:
        demand = cust_demands[customer]
        distances_from_customer = dist_matrix[customer]
        best_cost, best_route_index, best_position = float("inf"), None, None

        # 1. spots next to the nearest neighbours that are already in a route with room
        for neighbour in nearest_neighbours[customer]:
            route_index = route_of[neighbour]
            if route_index < 0 or loads[route_index] + demand > vehicle_cap:
                continue
            route = routes[route_index]
            neighbour_position = position_in_route[neighbour]
            for position in (neighbour_position, neighbour_position + 1):
                cost = insertion_cost(route, position, distances_from_customer)
                if cost < best_cost:
                    best_cost, best_route_index, best_position = cost, route_index, position

        # 2. no neighbour's route has room: try every spot in every route with room
        if best_route_index is None:
            for route_index, route in enumerate(routes):
                if loads[route_index] + demand > vehicle_cap:
                    continue
                for position in range(1, len(route)):
                    cost = insertion_cost(route, position, distances_from_customer)
                    if cost < best_cost:
                        best_cost, best_route_index, best_position = cost, route_index, position

        # 3. fits in no route: open a new one if a vehicle is free, otherwise give up
        if best_route_index is None:
            if len(routes) >= num_vehicles or demand > vehicle_cap:
                return None
            routes.append([0, customer, 0])
            loads.append(demand)
            index_route(len(routes) - 1)
            continue

        routes[best_route_index].insert(best_position, customer)
        loads[best_route_index] += demand
        index_route(best_route_index)

    return routes


def ruin_and_recreate(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                      nearest_neighbours, num_remove, rng):
    '''
    Removes num_remove nearby customers from the routes and puts them back at their cheapest spots.
    Returns (new routes, removed_customers), or (None, removed_customers) if a customer could not be placed
    '''
    ruined_routes, removed_customers = ruin(routes, nearest_neighbours, num_remove, rng)
    new_routes = recreate(ruined_routes, removed_customers, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                          nearest_neighbours, rng)
    return new_routes, removed_customers # we need removed_customers so that we can skip 2-opt on routes that have no changes in the customers
