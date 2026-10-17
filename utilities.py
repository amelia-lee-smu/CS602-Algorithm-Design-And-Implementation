def is_feasible(routes, cust_demands, vehicle_capacity, num_vehicles):
    '''Checks if the proposed solution is feasible or not.'''
    # number of routes does not exceed number of vehicles
    if len(routes) > num_vehicles:
        print(f"{len(routes)} routes used but only {num_vehicles} vehicles")
        return False

    num_customers = len(cust_demands) - 1 # remove depot

    visited = []
    for i, route in enumerate(routes):
        # every route must start and end at depot
        if route[0] != 0 or route[-1] != 0:
            print(f"route {i} does not start and end at depot: {route}")
            return False

        # add customers to visited list
        customers = route[1:-1]
        visited.extend(customers)

        # total demand of each route must not exceed vehicle_capacity
        load = sum(cust_demands[c] for c in customers)
        if load > vehicle_capacity:
            print(f"route {i} load {load} exceeds vehicle_capacity {vehicle_capacity}")
            return False

    # every customer must be visited exactly once
    if sorted(visited) != list(range(1, num_customers + 1)):
        print("customers not visited exactly once")
        return False

    return True

def calculate_cost(routes, dist_matrix):
    '''Calculates the total costs based on the routes and distance matrix.'''
    total_cost = 0

    # iteratively adds the cost
    for route in routes:
        for i in range(len(route) - 1):
            total_cost += dist_matrix[route[i]][route[i + 1]]

    return total_cost
