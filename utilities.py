'''
Shared helpers: reading instances, cost, feasibility and the CSV log.
'''

import csv
import os
import time


# ---------------------------------------------------------------------------
# Reading instances and solutions
# ---------------------------------------------------------------------------

def to_number(text):
    '''Converts "5" to 5 and "2.5" to 2.5'''
    value = float(text)
    return int(value) if value.is_integer() else value


def read_instance(path):
    '''
    Reads an instance file in the project format: a line of demands (depot first),
    then the distance matrix, then "Capacity: ..." and "Vehicles: ...".
    Returns (cust_demands, dist_matrix, vehicle_cap, num_vehicles)
    '''
    with open(path) as instance_file:
        lines = [line.strip() for line in instance_file if line.strip()]

    vehicle_cap = num_vehicles = None
    number_rows = []
    for line in lines:
        label = line.lower()
        if label.startswith("capacity"):
            vehicle_cap = to_number(line.split(":")[1])
        elif label.startswith("vehicles"):
            num_vehicles = int(line.split(":")[1])
        else:
            number_rows.append([to_number(value) for value in line.split()])

    cust_demands, dist_matrix = number_rows[0], number_rows[1:]
    return cust_demands, dist_matrix, vehicle_cap, num_vehicles


def read_solution(path):
    '''
    Reads a solution file: the cost on the first line, then one route per line.
    Returns (cost, routes)
    '''
    with open(path) as solution_file:
        lines = [line.strip() for line in solution_file if line.strip()]
    cost = to_number(lines[0])
    routes = [[int(stop) for stop in line.split()] for line in lines[1:]]
    return cost, routes


# ---------------------------------------------------------------------------
# Cost and feasibility
# ---------------------------------------------------------------------------

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


def instance_infeasibility(cust_demands, vehicle_capacity, num_vehicles):
    '''
    Checks whether the instance can ever be solved, whatever the routes:
    1. no single customer may need more than vehicle_capacity
    2. the total demand must fit in num_vehicles x vehicle_capacity
    Returns "" if it may be solvable, otherwise why it cannot be, e.g. "customer 12 demand 120 > capacity 100"
    '''
    heaviest = max(cust_demands)
    if heaviest > vehicle_capacity:
        return f"customer {cust_demands.index(heaviest)} demand {heaviest} > capacity {vehicle_capacity}"

    total_demand = sum(cust_demands)
    if total_demand > num_vehicles * vehicle_capacity:
        return f"total demand {total_demand} > {num_vehicles} vehicles x capacity {vehicle_capacity}"

    return ""


# ---------------------------------------------------------------------------
# CSV log (one file per batch of test cases)
# Only used for testing: the solver never reads anything back from the log,
# so every line tagged "# LOG" in route_first_cluster_second.py can be deleted.
# ---------------------------------------------------------------------------

# Columns of the log file, in order
LOG_COLUMNS = [
    "instance",       # test case name, e.g. A-n32-k5
    "instance_type",  # small / medium / large
    "time_s",         # seconds since the run started
    "step_time_s",    # seconds since the previous row, i.e. how long this step took
    "iteration",      # 0 = setup, then 1, 2, 3, ... (one giant tour, or one ruin and recreate)
    "step",           # what just finished (see LOG_STEPS)
    "cost",           # cost right after this step (empty if the step failed)
    "best_cost",      # best feasible cost so far (empty if there is none yet)
    "new_best",       # yes if this step's routes were saved as the new best
    "routes",         # routes used / vehicles available, e.g. 5/5
    "detail",         # why a step failed, or facts about the instance
]

# Values of the step column. Each row is written when its step FINISHES, so if the process
# was terminated, the step after the last row is the one that was still running.
# An iteration runs: giant_tour -> split / split_soft (or a failure) -> relocate_swap -> two_opt
#              or:   ruin_recreate (or recreate_failed) -> relocate_swap -> two_opt
LOG_STEPS = {
    "start":                "setup finished; detail has the seed, size and tightness of the instance",
    "instance_infeasible":  "the instance can never be solved (detail says why); the run stops",
    "giant_tour":           "giant tour built and 2-opted; cost = its length as a round trip from the depot",
    "split":                "tour cut into routes within capacity",
    "split_soft":           "split found no cut, so routes were allowed to overload, then repaired",
    "split_soft_no_cut":    "FAILED: no cut into num_vehicles routes, even with overloaded routes allowed",
    "repair_failed":        "FAILED: routes still overloaded after repair (detail: how much)",
    "partition_infeasible": "FAILED: routes broke a rule, which is a bug (detail: which rule)",
    "ruin_recreate":        "customers removed from the best solution and put back",
    "recreate_failed":      "FAILED: a removed customer fit in no route",
    "relocate_swap":        "customers moved between routes",
    "two_opt":              "customers reordered inside each route",
}


def open_log(log_info):
    '''
    log_info: None (logging off), or a dict with
        log_path, instance, instance_type (from the batch tester)
    If log_info is None, return None, so that nothing is logged.
    Otherwise:
    1. Record the start time (time.monotonic), used for time_s and step_time_s
    2. Create the logs folder if it does not exist
    3. Open log_path in append mode
    4. Write the header row (LOG_COLUMNS) only if the file is new or empty
    Returns a log handle holding the start time, the time of the last row, the open file, the CSV writer and log_info
    '''
    if log_info is None:
        return None

    start_time = time.monotonic()
    log_path = log_info["log_path"]

    folder = os.path.dirname(log_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    is_new_file = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if is_new_file:
        writer.writerow(LOG_COLUMNS)
        log_file.flush()

    return {"start_time": start_time, "last_time": start_time, "file": log_file, "writer": writer,
            "info": log_info}


def log_checkpoint(log, iteration, step, cost, best_cost, new_best, num_routes, num_vehicles, detail=""):
    '''
    Write one row to the CSV log. Does nothing if log is None.
    cost: None if the step failed; best_cost: infinity if there is no feasible solution yet;
    num_routes: None if there are no routes to count.
    Flushes straight away, so the row is kept if the process is terminated.
    '''
    if log is None:
        return

    now = time.monotonic()
    step_time = now - log["last_time"]
    log["last_time"] = now

    info = log["info"]
    log["writer"].writerow([
        info["instance"],
        info["instance_type"],
        f"{now - log['start_time']:.3f}",
        f"{step_time:.3f}",
        iteration,
        step,
        "" if cost is None else f"{cost:.2f}",
        "" if best_cost == float("inf") else f"{best_cost:.2f}",
        "yes" if new_best else "",
        "" if num_routes is None else f"{num_routes}/{num_vehicles}",
        detail,
    ])
    log["file"].flush()     # written to the OS straight away, so it survives the process being terminated


def infeasibility_reason(routes, cust_demands, vehicle_capacity, num_vehicles):
    '''
    Which rule the solution breaks, as text, e.g. "customer 12 visited 2 times", or "" if it is feasible.
    Slower than is_feasible (it finds the exact problem), so it is only used for the log, after a failure.
    '''
    if routes is None:
        return "no routes"
    if len(routes) > num_vehicles:
        return f"{len(routes)} routes but only {num_vehicles} vehicles"

    num_customers = len(cust_demands) - 1
    times_visited = [0] * (num_customers + 1)
    for index, route in enumerate(routes):
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            return f"route {index} does not start and end at the depot"
        load = 0
        for customer in route[1:-1]:
            if not 1 <= customer <= num_customers:
                return f"route {index} has an unknown stop {customer}"
            times_visited[customer] += 1
            load += cust_demands[customer]
        if load > vehicle_capacity:
            return f"route {index} load {load} > capacity {vehicle_capacity}"

    for customer in range(1, num_customers + 1):
        if times_visited[customer] != 1:
            return f"customer {customer} visited {times_visited[customer]} times"
    return ""


def failure_detail(step, failed_routes, cust_demands, vehicle_capacity, num_vehicles):
    '''
    The detail column for a failed partition (see LOG_STEPS), worked out from the routes that failed.
    Only called for the log, after a failure.
    '''
    if step == "repair_failed":
        loads = [sum(cust_demands[customer] for customer in route) for route in failed_routes]
        overload = sum(load - vehicle_capacity for load in loads if load > vehicle_capacity)
        num_overloaded = sum(1 for load in loads if load > vehicle_capacity)
        return (f"overload left {overload} over {num_overloaded} routes "
                f"(heaviest route {max(loads)}/{vehicle_capacity})")
    if step == "partition_infeasible":
        return infeasibility_reason(failed_routes, cust_demands, vehicle_capacity, num_vehicles)
    if step == "split_soft_no_cut":
        return f"tour order cannot be cut into {num_vehicles} routes, even with overloaded routes allowed"
    return ""
