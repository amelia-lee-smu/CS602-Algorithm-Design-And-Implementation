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


# ---------------------------------------------------------------------------
# CSV log (one file per batch of test cases)
# ---------------------------------------------------------------------------

# Columns of the log file, in order
LOG_COLUMNS = [
    "run_id",         # when the batch started, e.g. 2026-10-25_18-30-05
    "instance",       # test case name, e.g. A-n32-k5
    "instance_type",  # small / medium / large
    "seed",           # random seed of the run
    "time_s",         # seconds since the run started (3 decimal places)
    "iteration",      # giant tour number: 1, 2, 3, ...
    "step",           # checkpoint name: split / split_soft / relocate_swap / two_opt
    "cost",           # cost right after this step (2 decimal places)
    "best_cost",      # best cost so far
    "new_best",       # yes / no: this step produced a new best, which was saved
    "num_routes",     # number of routes in this solution
]


def open_log(log_info):
    '''
    log_info: None (logging off), or a dict with
        log_path, run_id, instance, instance_type (from the batch tester)
        and seed (added by run_rfcs)
    If log_info is None, return None, so that log_checkpoint does nothing.
    Otherwise:
    1. Record the start time (time.monotonic), used for time_s in every row
    2. Create the logs folder if it does not exist
    3. Open log_path in append mode
    4. Write the header row (LOG_COLUMNS) only if the file is new or empty
    Returns a log handle holding the start time, the open file, the CSV writer and log_info
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

    return {"start_time": start_time, "file": log_file, "writer": writer, "info": log_info}


def log_checkpoint(log, iteration, step, cost, best_cost, new_best, num_routes):
    '''
    Write one row to the CSV log. Does nothing if log is None.
    1. Fill run_id, instance, instance_type and seed from log_info
    2. time_s = time.monotonic() - the start time in log, rounded to 3 decimal places;
       round all costs to 2
    3. Write yes / no for new_best
    4. Flush straight away, so the row is kept if the process is terminated
    '''
    if log is None:
        return

    info = log["info"]
    log["writer"].writerow([
        info["run_id"],
        info["instance"],
        info["instance_type"],
        info["seed"],
        f"{time.monotonic() - log['start_time']:.3f}",
        iteration,
        step,
        f"{cost:.2f}",
        f"{best_cost:.2f}",
        "yes" if new_best else "no",
        num_routes,
    ])
    log["file"].flush()     # written to the OS straight away, so it survives the process being terminated
