'''
Shared helper functions.
'''

SEED = 42                    # random seed, so every run can be reproduced


def calculate_cost(dist_matrix, routes):
    '''
    Calculates the total costs based on the routes and distance matrix.
    '''
    total_cost = 0

    # iteratively adds the cost
    for route in routes:
        for i in range(len(route) - 1):
            total_cost += dist_matrix[route[i]][route[i + 1]]

    return total_cost


def is_feasible(routes, cust_demands, vehicle_capacity, num_vehicles):
    '''
    Returns True only if the solution meets every rule:
    1. The number of routes is at most num_vehicles
    2. Every route starts and ends at the depot (vertex 0)
    3. Every customer is visited exactly once across all routes
    4. Total demand of each route must not exceed vehicle_capacity
    '''
    # number of routes does not exceed number of vehicles
    if len(routes) > num_vehicles:
        # print(f"{len(routes)} routes used but only {num_vehicles} vehicles")
        return False

    num_customers = len(cust_demands) - 1 # remove depot

    visited = []
    for i, route in enumerate(routes):
        # every route must start and end at depot
        if route[0] != 0 or route[-1] != 0:
            # print(f"route {i} does not start and end at depot: {route}")
            return False

        # add customers to visited list
        customers = route[1:-1]
        visited.extend(customers)

        # total demand of each route must not exceed vehicle_capacity
        load = sum(cust_demands[c] for c in customers)
        if load > vehicle_capacity:
            # print(f"route {i} load {load} exceeds vehicle_capacity {vehicle_capacity}")
            return False

    # every customer must be visited exactly once
    if sorted(visited) != list(range(1, num_customers + 1)):
        # print("customers not visited exactly once")
        return False

    return True


def check_and_update_new_best(routes, state):
    '''
    Called when a potentially better solution has been found, so that it can be updated. 
    1. Recompute the cost from the routes.
    2. If it is cheaper than the best so far, check it with is_feasible,
       then save it to best["solution"] as one (cost, routes) write.
    Returns the cost of routes.
    '''
    # recompute the cost from the routes
    cost = calculate_cost(state["dist_matrix"], routes)

    # save if new best
    if cost < state["best_cost"] and is_feasible(routes, state["cust_demands"], state["vehicle_cap"], state["num_vehicles"]):
        best_routes = [route[:] for route in routes]   # copy, so later steps cannot change the saved routes
        state["best_cost"] = cost                       # local copy: fast to compare against
        state["best"]["solution"] = (cost, best_routes) # shared copy: what run_algo returns, in one write

    # Note: there is no need to return state since python just updates the dict directly
    return cost
