import random

from utilities import *
from optimisations import *
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


def run_rfcs_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, num_giant_tours=25, seed=SEED, time_limit=None):
    '''
    Implementation of the Route first, cluster second algorithm (Beasley, 1983).
    For each sample of giant tour:
    (1) generate a random giant tour (excluding depot) and improve it with 2-opt
    (2) partition the giant tour into individual routes
    (3) do 3-opt to optimize each individual route
    '''
    best_cost = INF                                     # replace when a lower cost is found
    best_routes = None                                  # replace when a better set of routes are found
    rng = random.Random(seed)                           # random generator with seed
    customers = list(range(1, len(cust_demands)))       # get the customer indexes without the depot

    for _ in range(num_giant_tours):

        # (1) route first: random giant tour, improved with 2-opt
        tour = customers[:]                             # make a copy of the customer list so that it wont be shuffled
        rng.shuffle(tour)                               # shuffle the customer order
        tour = two_opt(tour, dist_matrix)               # improve the giant tour using 2-opt

        # (2) cluster second: partition the giant tour
        routes = partition_giant_route(tour, cust_demands, dist_matrix, vehicle_capacity)

        # (3) 3-optimise each route
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
