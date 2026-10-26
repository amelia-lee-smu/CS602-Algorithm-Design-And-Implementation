from optimisations import MAX_PENALTY_DOUBLINGS, cheapest_insertion, repair_overloads, vehicle_reduction
from utilities import check_and_update_new_best


def run_clarke_wright(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state):
    '''
    Clarke-Wright Savings Algorithm (Parallel Version for CVRP).
    Calculates savings for all customer pairs, sorts them in descending order,
    and merges single-customer routes into feasible multi-customer routes.
    '''
    num_customers = len(cust_demands) - 1
    
    # 1. Start with single-customer routes: depot (0) -> i -> depot (0)
    routes = [[0, i, 0] for i in range(1, num_customers + 1)]
    loads = [cust_demands[i] for i in range(1, num_customers + 1)]
    
    # Track the route index for each customer
    route_of = {i: i - 1 for i in range(1, num_customers + 1)}

    # 2. Calculate savings: S_ij = d(0, i) + d(0, j) - d(i, j)
    savings = []
    for i in range(1, num_customers + 1):
        for j in range(i + 1, num_customers + 1):
            s = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]
            savings.append((s, i, j))

    # Sort savings in descending order
    savings.sort(key=lambda x: x[0], reverse=True)

    # 3. Merge routes according to savings
    for s, i, j in savings:
        r_i = route_of[i]
        r_j = route_of[j]

        # Ignore if both customers are already on the same route
        if r_i == r_j:
            continue

        route_i = routes[r_i]
        route_j = routes[r_j]

        # Ignore if the combined loads exceed the vehicle capacity
        if loads[r_i] + loads[r_j] > vehicle_cap:
            continue

        # Check merge eligibility (i and j must be interior/exterior endpoints next to depot 0)
        # Case 1: i is at end of route_i, j is at start of route_j 
        # (0 -> ... -> i -> 0  +  0 -> j -> ... -> 0)
        if route_i[-2] == i and route_j[1] == j:
            merged = route_i[:-1] + route_j[1:]
        # Case 2: j is at end of route_j, i is at start of route_i 
        # (0 -> ... -> j -> 0  +  0 -> i -> ... -> 0)
        elif route_j[-2] == j and route_i[1] == i:
            merged = route_j[:-1] + route_i[1:]
        # Case 3: i is at end of route_i, j is at end of route_j (reverse route_j's customers)
        # (0 -> ... -> i -> j -> ... -> 0)
        elif route_i[-2] == i and route_j[-2] == j:
            merged = route_i[:-1] + route_j[1:-1][::-1] + [0]
        # Case 4: i is at start of route_i, j is at start of route_j (reverse route_j's customers, put them first)
        # (0 -> ... -> j -> i -> ... -> 0)
        elif route_i[1] == i and route_j[1] == j:
            merged = [0] + route_j[1:-1][::-1] + route_i[1:]
        # ignore if the 4 merge conditions are not satisfied
        else:
            continue

        # Perform the route merge
        routes[r_i] = merged             # 1. i's route becomes the merged route
        routes[r_j] = []                 # 2. j's route is emptied
        loads[r_i] += loads[r_j]         # 3. i's route now carries both loads
        loads[r_j] = 0                   # 4. j's emptied route carries nothing

        # Update customer route lookup mapping
        for cust in merged[1:-1]:
            route_of[cust] = r_i

    # Filter out empty routes
    final_routes = [r for r in routes if len(r) > 2]

    # 4. If too many vehicles used, try cutting down
    if len(final_routes) > num_vehicles:

        # remove the cheapest routes, until num_vehicles is reached
        kept_routes, removed_customers = vehicle_reduction(final_routes, cust_demands, num_vehicles)
        
        # do cheapest insertion to slot customers in (may overload routes)
        overloaded_routes = cheapest_insertion(kept_routes, removed_customers, cust_demands, dist_matrix, vehicle_cap)

        # repair the overloaded routes by moving customers between nearby routes
        final_routes = repair_overloads(overloaded_routes, cust_demands, dist_matrix, vehicle_cap,
                                        nearest_neighbours, rng, overload_penalty,
                                        MAX_PENALTY_DOUBLINGS)
        
        # do nothing if no feasible results
        if final_routes is None:
            return

    # save the solution only if it is feasible
    check_and_update_new_best(final_routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, state)
