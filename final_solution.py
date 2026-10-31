import heapq
import multiprocessing
import random

from clark_wright_savings import run_clarke_wright
from optimisations import MAX_PENALTY_DOUBLINGS, repair_overloads, two_opt
from route_first_cluster_second import run_rfcs
from utilities import SEED, check_and_update_new_best, is_feasible

# Time limit in seconds for each instance type
TIME_LIMITS = {
    "small": 30,
    "medium": 60,
    "large": 180,
}
# Stop at 95% of the time limit, leaving time to write the output
TIME_LIMIT_FRACTION = 0.95

# nearest neighbours used by 2-opt and relocate/swap
NUM_NEIGHBOURS = 30


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
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
# Packing by demand
# ---------------------------------------------------------------------------
def pack_by_demand(cust_demands, dist_matrix, vehicle_cap, num_vehicles):
    '''
    Packs customers into at most num_vehicles routes by demand only:
    1. biggest demand first, each customer goes into the fullest route it still fits in
    2. if it fits in none, it goes into the emptiest route (overloaded, for repair_overloads to fix)
    3. each route is ordered by nearest neighbour from the depot, then 2-opt
    Returns the routes (may be overloaded)
    '''
    num_customers = len(cust_demands) - 1
    customers_in_route = [[] for _ in range(num_vehicles)]
    loads = [0] * num_vehicles

    # 1-2. best fit decreasing
    for customer in sorted(range(1, num_customers + 1), key=cust_demands.__getitem__, reverse=True):
        demand = cust_demands[customer]
        best_route_index, least_room_left = None, float("inf")
        for route_index in range(num_vehicles):
            room_left = vehicle_cap - loads[route_index] - demand
            if 0 <= room_left < least_room_left:
                best_route_index, least_room_left = route_index, room_left

        if best_route_index is None:    # fits nowhere: overload the emptiest route
            best_route_index = min(range(num_vehicles), key=loads.__getitem__)

        customers_in_route[best_route_index].append(customer)
        loads[best_route_index] += demand

    # 3. order each route, skipping unused vehicles
    routes = []
    for customers in customers_in_route:
        if not customers:
            continue
        route = [0]
        unvisited = set(customers)
        current_customer = 0
        while unvisited:
            current_customer = min(unvisited, key=dist_matrix[current_customer].__getitem__)
            unvisited.remove(current_customer)
            route.append(current_customer)
        route.append(0)
        routes.append(two_opt(route, dist_matrix))

    return routes


def run_packing(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state):
    '''
    Packs by demand, repairs only if a route is overloaded, and saves the routes if feasible.
    '''
    routes = pack_by_demand(cust_demands, dist_matrix, vehicle_cap, num_vehicles)

    # repair overload if needed
    if not is_feasible(routes, cust_demands, vehicle_cap, num_vehicles):
        routes = repair_overloads(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours,
                                  rng, overload_penalty, MAX_PENALTY_DOUBLINGS)
        
    # do nothing if no feasible results
    if routes is None:
        return

    check_and_update_new_best(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, state)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------
def find_solution(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best):
    # the state dict that will be passed around the algos.
    state = {
        "best": best,
        "best_cost": best["solution"][0],
    }

    # built once, used by all three steps
    rng = random.Random(SEED)                                                   # one random generator, so every run can be reproduced
    nearest_neighbours = build_nearest_neighbours(dist_matrix)                  # each customer's nearest neighbours
    overload_penalty = starting_overload_penalty(cust_demands, dist_matrix)     # cost per unit of overload, for repairs

    # 1. Packing by demand: a very fast solution (highest chance of feasible solution)
    run_packing(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state)

    # 2. Clarke-Wright savings: a better first solution, saved only if cheaper than the packing
    run_clarke_wright(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state)

    # 3. Route first, cluster second (RFCS): search for a better solution with the remaining time
    run_rfcs(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state)

    best_cost, best_routes = best["solution"]
    return best_cost, best_routes


def run_algo(cust_demands, dist_matrix, vehicle_cap, num_vehicles, instance_type):
    TIMEOUT = TIME_LIMIT_FRACTION * TIME_LIMITS[instance_type]

    with multiprocessing.Manager() as manager:
        # Saved best solution, in the format (cost, list of routes)
        best = manager.dict(solution=(float("inf"), None))

        # Run the algo
        process = multiprocessing.Process(
            target=find_solution, 
            args=(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best)
        )
        process.start()
        process.join(timeout=TIMEOUT) # set the time limit for the run

        # Terminate the process when the time limit is hit
        if process.is_alive():
            print(f"Code terminated at {TIMEOUT} seconds.")
            process.terminate()  # Forces the process to close
            process.join()       # Cleans up the process resources

        best_cost, best_routes = best["solution"]

    return best_cost, best_routes


# if __name__ == "__main__":
#     from test_cases.sample_from_instructions import (
#         cust_demands,
#         dist_matrix,
#         vehicle_capacity,
#         num_vehicles,
#     )

#     best_cost, best_routes = run_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, "small")
#     print(best_cost)
#     print(best_routes)
