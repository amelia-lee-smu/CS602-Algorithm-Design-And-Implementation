'''
Improvement steps shared by Clarke-Wright and RFCS.
'''
# TODO: unchecked
def two_opt(path, dist_matrix):
    '''
    Improves the order of a path by reversing segments, trying every pair of edges,
    until no reversal makes it shorter. Best for short paths such as a single route.
    The first and last items stay fixed. The path is changed in place.
    Returns the improved path
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
    Improves the order of a path by reversing segments, until no reversal makes it shorter.
    Each customer is only paired with its nearest neighbours.
    its first and last items stay fixed. The path is changed in place.
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


# TODO: not sure if needed
def three_opt(route, dist_matrix):
    '''
    Improve the order of one route by removing 3 edges and reconnecting
    the 3 segments in the cheapest way.
    Repeat until no reconnection lowers the cost.
    Routes are short (one vehicle's customers), so trying every combination is fine.
    Returns the improved route
    '''
    pass


# TODO: unchecked
def repair_overloads(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                     penalty, max_doublings):
    '''
    Removes the overloads left by split_with_vehicle_overload.
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

    # split_with_vehicle_overload may already return routes within capacity: no repair needed
    if not has_overload(routes):
        return routes

    for _ in range(max_doublings + 1):
        routes = relocate_and_swap(routes, cust_demands, dist_matrix, vehicle_cap,
                                   nearest_neighbours, rng, penalty)
        if not has_overload(routes):
            return routes
        penalty *= 2    # local to this call: the caller's starting penalty is unchanged

    return None


# TODO: unchecked
def relocate_and_swap(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng, penalty=None):
    '''
    Moves customers between routes until no move lowers the cost:
    - relocate: move a customer next to one of its nearest neighbours on another route
    - swap: exchange a customer with one of its nearest neighbours on another route
    penalty=None: every route must stay within vehicle_cap (normal improvement).
    penalty given: routes may be overloaded, and cost = distance + penalty * total overload (repair).
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
    customers = [customer for route in routes for customer in route[1:-1]]
    improved = True
    while improved:
        improved = False
        rng.shuffle(customers)
        for customer in customers:
            if try_relocate(customer) or try_swap(customer):
                improved = True

    return [route for route in routes if len(route) > 2]    # drop routes left empty


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------
# TODO: not sure if needed
def insertion_cost(route, position, customer, dist_matrix):
    '''
    Extra distance from inserting customer between route[position - 1] and route[position]:
    d(prev, customer) + d(customer, next) - d(prev, next)
    '''
    pass


# TODO: not sure if needed
def cheapest_insertion(routes, loads, customer, cust_demands, dist_matrix, vehicle_cap):
    '''
    Find the cheapest position for customer in any route that still has room.
    1. Skip routes where loads[r] + demand of customer > vehicle_cap
    2. For every position in the remaining routes, compute insertion_cost
    3. Insert customer at the cheapest position and update loads
    Returns True if the customer was inserted, False if it fits in no route
    '''
    pass


# ---------------------------------------------------------------------------
# Ruin and recreate
# ---------------------------------------------------------------------------
# TODO: not sure if needed
def ruin(routes, nearest_neighbours, num_remove, rng):
    '''
    Remove a group of customers that are close together.
    1. Pick a random customer
    2. Remove it and its nearest neighbours until num_remove customers are removed
    3. Delete any route left with no customers
    Returns (routes, removed customers)
    '''
    pass


# TODO: not sure if needed
def recreate(routes, removed, cust_demands, dist_matrix, vehicle_cap, num_vehicles, rng):
    '''
    Put the removed customers back, one at a time.
    1. Shuffle the removed customers, so each attempt rebuilds differently
    2. For each customer, use cheapest_insertion
    3. If it fits in no route and fewer than num_vehicles routes are used,
       open a new route [0, customer, 0]
    4. If it still cannot be placed, give up on this attempt
    Returns the new routes, or None if a customer could not be placed
    '''
    pass


# TODO: not sure if needed
def ruin_and_recreate(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                      nearest_neighbours, num_remove, rng):
    '''
    1. Copy routes, so the input solution is not changed
    2. ruin the copy
    3. recreate it
    Returns the new routes, or None if recreate failed
    '''
    pass
