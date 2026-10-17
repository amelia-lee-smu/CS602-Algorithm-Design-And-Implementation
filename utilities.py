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

# Functions for reading test case instances
def to_num(x):
    v = float(x)
    return int(v) if v.is_integer() else v


def read_instance(path):
    '''Reads the exact test case path. Finds Capacity/Vehicles by label'''
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    capacity = vehicles = None
    rows = []
    for line in lines:
        low = line.lower()
        if low.startswith('capacity'):
            capacity = to_num(line.split(':')[1])
        elif low.startswith('vehicles'):
            vehicles = int(line.split(':')[1])
        else:
            rows.append([to_num(x) for x in line.split()])
    demands, matrix = rows[0], rows[1:]
    return demands, matrix, capacity, vehicles

def read_solution(path):
    '''Reads solution of corresponding test case, returns cost and best routes'''
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    cost = to_num(lines[0])
    routes = [[int(x) for x in line.split()] for line in lines[1:]]
    return cost, routes

def score(opt, cost, sml, t, verbose=False):
    '''Scores best traversal output against best path found so far. 
    Returns values from the time component, cost component, the combined term from earlier 2 and the final score. 
    Terms are returned separately to help identify bottleneck
    opt = optimal traversal cost
    cost = cost of proposed traversal
    sml = input "small", "medium", "large" for classification
    t = time taken for proposed traversal to run
    '''
    timing = {'small': 30, 'medium': 60, 'large': 180}
    t_l = timing[sml]
    if t > t_l * 1.5:
        return None, None, None, 0.25
    time_term = 2 ** (max(0, t / t_l - 1))
    cost_term = cost / opt
    combined_term = 0.5 * (time_term + cost_term)
    if verbose:
        print('time term: ', time_term)
        print('cost term: ', cost_term)
        print('combined term: ', combined_term)
    return time_term, cost_term, combined_term, 1 / combined_term