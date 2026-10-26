'''
Route first, cluster second (Beasley, 1983), with modifications.
'''
from optimisations import *
from utilities import *


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
NN_CHOICES = 3                      # giant tour picks at random among the 3 nearest unvisited customers
RANDOMISED_RESTART_AFTER = 1000     # use a new giant tour after this many consecutive ruin and recreate iterations without a cost improvement



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
def partition_giant_tour(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                         nearest_neighbours, rng, penalty):
    '''
    Cuts the giant tour into routes that should be feasible:
    - tries split_within_vehicle_capacity first
    - if it finds no cut that fits within num_vehicles, uses split_with_overload_and_repair
    The routes are not checked here: run_rfcs checks them with is_feasible before using them.

    Returns (routes, method). method says which approach worked, or why it failed:
        (routes, "split") or (routes, "split_soft")   routes from that approach
        (None, "split_soft_no_cut")                   no cut, even with overload allowed
        (None, "repair_failed")                       the repair after split_soft failed
    '''
    # 1. exact split: fast, no repair needed
    routes = split_within_vehicle_capacity(tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles)
    method = "split"

    # 2. no cut fits: split with overload allowed, then repair
    if routes is None:
        routes, method = split_with_overload_and_repair(
            tour, cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, rng, penalty)

    return routes, method


# ---------------------------------------------------------------------------
# Improvement steps
# ---------------------------------------------------------------------------
def improve_between_routes(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                           customers_to_check=None):
    '''
    Moves and swaps customers between nearby routes, keeping every route within vehicle_cap.
    customers_to_check: only try moves for these customers (None = every customer).
    Returns the improved routes.
    '''
    return relocate_and_swap(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                             customers_to_check=customers_to_check)


def improve_within_routes(routes, dist_matrix, customers_to_check=None):
    '''
    Reorders the customers inside routes with plain 2-opt (loads do not change, so routes stay feasible).
    customers_to_check: only routes that contain these customers are optimised using 2-opt.
    Returns the improved routes.
    '''
    changed_customers = None if customers_to_check is None else set(customers_to_check)

    for route in routes:
        if changed_customers is None or not changed_customers.isdisjoint(route):
            two_opt(route, dist_matrix)
    return routes


def improve_feasible_routes(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, rng,
                            state, customers_to_check=None):
    '''
    Improves feasible routes (from partition_giant_tour, or from ruin and recreate):
    1. between routes: relocate and swap customers between nearby routes
    2. within routes: 2-opt on each route
    customers_to_check: only work around these customers, e.g. the ones ruin and recreate moved. otherwise, None.
    state: from run_rfcs; check_and_update_new_best is called after each step,
    so a new best is saved.
    Returns (improved routes, new cost)
    '''
    routes = improve_between_routes(routes, cust_demands, dist_matrix, vehicle_cap, nearest_neighbours, rng,
                                    customers_to_check)
    check_and_update_new_best(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, state)

    routes = improve_within_routes(routes, dist_matrix, customers_to_check)
    cost = check_and_update_new_best(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, state)     # also returns the cost, so it is not computed again

    return routes, cost


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_rfcs(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state):
    '''
    Outlines steps for the RFCS approach.
    '''
    # Setup
    num_customers = len(cust_demands) - 1
    current_cost = state["best_cost"]                                           # get the current best cost
    current_routes = state["best"]["solution"][1]                               # what ruin and recreate works on (None if no solution yet)

    # how many customers the ruin step removes
    max_remove = max(1, min(MAX_REMOVE, num_customers // 4))                    # cannot exceed 1/4 of the customers 
    min_remove = min(MIN_REMOVE, max_remove)                                    # must be smaller than max_remove
    iterations_without_improvement = 0                                          # ruin and recreate iterations since current last improved
    state["iteration"] = 0                                                      # RFCS loop count (read by test_case_logger.py)

    # Repeat until run_algo terminates the process
    while True:
        state["iteration"] += 1

        # build from a new randomised giant tour when there is no feasible solution yet,
        # or restart from one when ruin and recreate has stopped improving the current solution
        if current_routes is None or iterations_without_improvement >= RANDOMISED_RESTART_AFTER:
            # start with a new randomised giant tour, improved with optimised 2-opt
            tour = build_giant_tour(dist_matrix, nearest_neighbours, rng)

            # cut it into feasible routes
            routes, method = partition_giant_tour(tour, cust_demands, dist_matrix, vehicle_cap,
                                                  num_vehicles, nearest_neighbours, rng, overload_penalty)

            # no feasible routes from this tour: try a new tour if there is no solution yet
            if routes is None or not is_feasible(routes, cust_demands, vehicle_cap, num_vehicles):
                continue

            # if feasible, see if it is a new best
            check_and_update_new_best(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, state)
            iterations_without_improvement = 0          # reset attempts counter so that it goes to ruin and recreate in the next iteration

            # improve the routes, and make them the current solution, even if worse than the best
            current_routes, current_cost = improve_feasible_routes(routes, cust_demands, dist_matrix, vehicle_cap,
                                                                   num_vehicles, nearest_neighbours, rng, state)

        # use ruin and recreate to improve on the current solution
        else:
            num_remove = rng.randint(min_remove, max_remove)        # random number of customers to remove
            routes, removed_customers = ruin_and_recreate(current_routes, cust_demands, dist_matrix, vehicle_cap,
                                                          num_vehicles, nearest_neighbours, num_remove, rng)
            
            # if no feasible routes can be found, try ruin and recreate again
            if routes is None: # no need to check is_feasible, ruin and recreate guarantees a feasible result if routes are returned
                iterations_without_improvement += 1
                continue

            # if feasible, see if it is a new best
            check_and_update_new_best(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles, state)

            # improve on the current routes, and see there is a new best after each stage
            routes, cost = improve_feasible_routes(routes, cust_demands, dist_matrix, vehicle_cap, num_vehicles,
                                                   nearest_neighbours, rng, state, customers_to_check=removed_customers)

            # keep the result as the current solution only if it is cheaper
            if cost < current_cost - 1e-9:
                current_routes, current_cost = routes, cost
                iterations_without_improvement = 0
            else: 
                iterations_without_improvement += 1
